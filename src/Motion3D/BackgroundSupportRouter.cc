#include "Motion3D/BackgroundSupportRouter.h"

#include <Eigen/Eigenvalues>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace Motion3D {
namespace {

bool finiteDepth(float value, const BackgroundSupportRouter::Config& config)
{
    return std::isfinite(value) &&
        value >= config.min_depth &&
        value <= config.max_depth;
}

}  // namespace

BackgroundSupportRouter::BackgroundSupportRouter()
    : BackgroundSupportRouter(Config{}) {}

BackgroundSupportRouter::BackgroundSupportRouter(const Config& config)
    : config_(config)
{
    config_.ring_radius = std::max(1, config_.ring_radius);
    config_.sample_stride = std::max(1, config_.sample_stride);
    config_.max_support_points = std::max(1, config_.max_support_points);
    config_.min_support_points =
        std::clamp(config_.min_support_points, 1, config_.max_support_points);
    config_.min_depth = std::max(1e-4f, config_.min_depth);
    config_.max_depth = std::max(config_.min_depth, config_.max_depth);
    config_.max_depth_residual = std::max(1e-5f, config_.max_depth_residual);
    config_.depth_residual_scale = std::max(0.0f, config_.depth_residual_scale);
    config_.min_gradient = std::max(0.0f, config_.min_gradient);
    config_.flow_guard_radius = std::max(0, config_.flow_guard_radius);
}

cv::Mat BackgroundSupportRouter::toGrayFloat(const cv::Mat& image)
{
    if (image.empty()) {
        return {};
    }
    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        return {};
    }

    cv::Mat output;
    double scale = 1.0;
    if (gray.depth() == CV_8U) {
        scale = 1.0 / 255.0;
    } else if (gray.depth() == CV_16U) {
        scale = 1.0 / 65535.0;
    } else if (gray.depth() != CV_32F && gray.depth() != CV_64F) {
        return {};
    }
    gray.convertTo(output, CV_32F, scale);
    return output;
}

cv::Mat BackgroundSupportRouter::toBinaryMask(
    const cv::Mat& mask, const cv::Size& size)
{
    if (mask.empty() || mask.size() != size || mask.channels() != 1) {
        return {};
    }
    cv::Mat binary;
    if (mask.depth() == CV_8U) {
        binary = mask.clone();
    } else {
        mask.convertTo(binary, CV_8U);
    }
    cv::threshold(binary, binary, 0, 255, cv::THRESH_BINARY);
    return binary;
}

float BackgroundSupportRouter::median(std::vector<float> values)
{
    if (values.empty()) {
        return -1.0f;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5f * (values[middle - 1] + values[middle]);
    }
    return values[middle];
}

BackgroundSupportRouter::Result BackgroundSupportRouter::select(
    const cv::Mat& current_gray_or_bgr,
    const cv::Mat& depth_meters,
    const cv::Mat& dynamic_mask,
    const cv::Mat& static_mask,
    const cv::Mat& rendered_depth,
    const cv::Mat& flow_guard) const
{
    Result result;
    if (depth_meters.empty() || depth_meters.type() != CV_32FC1) {
        return result;
    }

    const cv::Mat gray = toGrayFloat(current_gray_or_bgr);
    if (gray.empty() || gray.size() != depth_meters.size()) {
        return result;
    }
    const cv::Mat dynamic = toBinaryMask(dynamic_mask, depth_meters.size());
    const cv::Mat statik = toBinaryMask(static_mask, depth_meters.size());
    if (dynamic.empty() || statik.empty()) {
        return result;
    }
    if (config_.require_rendered_depth &&
        (rendered_depth.empty() ||
         rendered_depth.type() != CV_32FC1 ||
         rendered_depth.size() != depth_meters.size())) {
        return result;
    }
    if (!rendered_depth.empty() &&
        (rendered_depth.type() != CV_32FC1 ||
         rendered_depth.size() != depth_meters.size())) {
        return result;
    }
    if (!flow_guard.empty() &&
        (flow_guard.size() != depth_meters.size() ||
         flow_guard.channels() != 1)) {
        return result;
    }

    result.dynamic_pixels = cv::countNonZero(dynamic);

    const int kernel_size = 2 * config_.ring_radius + 1;
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::Mat dilated;
    cv::dilate(dynamic, dilated, kernel);
    cv::Mat inverse_dynamic;
    cv::bitwise_not(dynamic, inverse_dynamic);
    cv::bitwise_and(dilated, inverse_dynamic, result.ring_mask);
    cv::bitwise_and(result.ring_mask, statik, result.ring_mask);
    result.ring_pixels = cv::countNonZero(result.ring_mask);
    result.support_mask =
        cv::Mat::zeros(depth_meters.size(), CV_8UC1);

    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
    cv::Mat gradient;
    cv::magnitude(grad_x, grad_y, gradient);

    cv::Mat guard;
    if (!flow_guard.empty()) {
        cv::Mat flow_binary = toBinaryMask(flow_guard, depth_meters.size());
        const int guard_kernel_size = 2 * config_.flow_guard_radius + 1;
        const cv::Mat guard_kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(guard_kernel_size, guard_kernel_size));
        cv::dilate(flow_binary, guard, guard_kernel);
    }

    std::vector<SupportPoint> candidates;
    candidates.reserve(static_cast<std::size_t>(result.ring_pixels));
    for (int y = 0; y < depth_meters.rows; y += config_.sample_stride) {
        const uchar* ring_row = result.ring_mask.ptr<uchar>(y);
        for (int x = 0; x < depth_meters.cols; x += config_.sample_stride) {
            if (ring_row[x] == 0) {
                continue;
            }
            if (!guard.empty() && guard.at<uchar>(y, x) != 0) {
                continue;
            }
            const float depth = depth_meters.at<float>(y, x);
            if (!finiteDepth(depth, config_)) {
                continue;
            }
            const float local_gradient = gradient.at<float>(y, x);
            if (!std::isfinite(local_gradient) ||
                local_gradient < config_.min_gradient) {
                continue;
            }

            float depth_residual = 0.0f;
            if (!rendered_depth.empty()) {
                const float rendered = rendered_depth.at<float>(y, x);
                if (!finiteDepth(rendered, config_)) {
                    continue;
                }
                depth_residual = std::abs(depth - rendered);
                const float tolerance = config_.max_depth_residual +
                    config_.depth_residual_scale * rendered;
                if (!std::isfinite(depth_residual) ||
                    depth_residual > tolerance) {
                    continue;
                }
            }

            const float residual_term =
                depth_residual /
                std::max(1e-5f, config_.max_depth_residual);
            const float score =
                local_gradient / (1.0f + residual_term);
            candidates.push_back(SupportPoint{
                cv::Point(x, y), depth, depth_residual,
                local_gradient, score});
        }
    }

    result.candidate_points = static_cast<int>(candidates.size());
    std::sort(candidates.begin(), candidates.end(),
              [](const SupportPoint& lhs, const SupportPoint& rhs) {
                  if (lhs.score != rhs.score) {
                      return lhs.score > rhs.score;
                  }
                  if (lhs.pixel.y != rhs.pixel.y) {
                      return lhs.pixel.y < rhs.pixel.y;
                  }
                  return lhs.pixel.x < rhs.pixel.x;
              });
    if (static_cast<int>(candidates.size()) >
        config_.max_support_points) {
        candidates.resize(static_cast<std::size_t>(
            config_.max_support_points));
    }
    result.points = candidates;
    result.selected_points = static_cast<int>(result.points.size());
    if (result.ring_pixels > 0) {
        result.support_ratio =
            static_cast<float>(result.selected_points) /
            static_cast<float>(result.ring_pixels);
    }
    std::vector<float> residuals;
    residuals.reserve(result.points.size());
    for (const SupportPoint& point : result.points) {
        result.support_mask.at<uchar>(point.pixel) = 255;
        residuals.push_back(point.depth_residual);
    }
    result.median_depth_residual = median(std::move(residuals));
    result.valid = result.selected_points >= config_.min_support_points;
    return result;
}

BackgroundSupportPoseRefiner::BackgroundSupportPoseRefiner(
    const Config& config, const Intrinsics& intrinsics)
    : config_(config),
      intrinsics_(intrinsics),
      router_(config.router)
{
    config_.max_iterations = std::max(1, config_.max_iterations);
    config_.huber_delta = std::max(1e-5f, config_.huber_delta);
    config_.min_inlier_ratio =
        std::clamp(config_.min_inlier_ratio, 0.0f, 1.0f);
    config_.min_rmse_improvement =
        std::clamp(config_.min_rmse_improvement, 0.0f, 0.99f);
    config_.max_translation_correction =
        std::max(0.0f, config_.max_translation_correction);
    config_.max_rotation_correction =
        std::max(0.0f, config_.max_rotation_correction);
    if (intrinsics_.fx <= 0.0f || intrinsics_.fy <= 0.0f) {
        throw std::invalid_argument(
            "BackgroundSupportPoseRefiner requires positive intrinsics");
    }
}

Eigen::Matrix3f BackgroundSupportPoseRefiner::skew(
    const Eigen::Vector3f& value)
{
    Eigen::Matrix3f result;
    result << 0.0f, -value.z(), value.y(),
              value.z(), 0.0f, -value.x(),
              -value.y(), value.x(), 0.0f;
    return result;
}

float BackgroundSupportPoseRefiner::huberWeight(
    float residual_norm, float delta)
{
    if (!std::isfinite(residual_norm) || residual_norm <= 0.0f) {
        return 0.0f;
    }
    return residual_norm <= delta ? 1.0f : delta / residual_norm;
}

bool BackgroundSupportPoseRefiner::project(
    const Eigen::Vector3f& point,
    const Intrinsics& intrinsics,
    const cv::Size& image_size,
    cv::Point2f& pixel)
{
    if (!point.allFinite() || point.z() <= 1e-5f) {
        return false;
    }
    pixel.x = intrinsics.fx * point.x() / point.z() + intrinsics.cx;
    pixel.y = intrinsics.fy * point.y() / point.z() + intrinsics.cy;
    return std::isfinite(pixel.x) && std::isfinite(pixel.y) &&
        pixel.x >= 1.0f && pixel.y >= 1.0f &&
        pixel.x < static_cast<float>(image_size.width - 2) &&
        pixel.y < static_cast<float>(image_size.height - 2);
}

float BackgroundSupportPoseRefiner::rmse(
    const std::vector<Correspondence>& correspondences,
    const Sophus::SE3f& relative_pose,
    float inlier_threshold,
    int* inliers)
{
    double squared_error = 0.0;
    int count = 0;
    for (const Correspondence& correspondence : correspondences) {
        const float residual = (
            relative_pose * correspondence.previous -
            correspondence.current).norm();
        if (!std::isfinite(residual)) {
            continue;
        }
        squared_error += static_cast<double>(residual) * residual;
        if (residual <= inlier_threshold) {
            ++count;
        }
    }
    if (inliers) {
        *inliers = count;
    }
    if (correspondences.empty()) {
        return std::numeric_limits<float>::infinity();
    }
    return static_cast<float>(
        std::sqrt(squared_error / correspondences.size()));
}

BackgroundSupportPoseRefiner::Result BackgroundSupportPoseRefiner::refine(
    const cv::Mat& previous_gray_or_bgr,
    const cv::Mat& current_gray_or_bgr,
    const cv::Mat& previous_depth_meters,
    const cv::Mat& current_depth_meters,
    const cv::Mat& previous_static_mask,
    const cv::Mat& current_static_mask,
    const cv::Mat& current_dynamic_mask,
    const cv::Mat& current_rendered_depth,
    const cv::Mat& current_flow_guard,
    const Sophus::SE3f& parent_relative_pose) const
{
    Result result;
    result.refined_relative_pose = parent_relative_pose;
    if (!parent_relative_pose.matrix().allFinite() ||
        previous_depth_meters.empty() ||
        current_depth_meters.empty() ||
        previous_depth_meters.type() != CV_32FC1 ||
        current_depth_meters.type() != CV_32FC1 ||
        previous_depth_meters.size() != current_depth_meters.size() ||
        previous_static_mask.empty() ||
        current_static_mask.empty()) {
        return result;
    }
    if (previous_static_mask.size() != previous_depth_meters.size() ||
        current_static_mask.size() != current_depth_meters.size()) {
        return result;
    }

    const auto routed = router_.select(
        current_gray_or_bgr, current_depth_meters, current_dynamic_mask,
        current_static_mask, current_rendered_depth, current_flow_guard);
    result.routed_support = routed.selected_points;
    if (routed.selected_points < config_.router.min_support_points) {
        return result;
    }

    const cv::Size image_size = current_depth_meters.size();
    std::vector<Correspondence> correspondences;
    correspondences.reserve(routed.points.size());
    for (const auto& support : routed.points) {
        const int x = support.pixel.x;
        const int y = support.pixel.y;
        const float current_depth =
            current_depth_meters.at<float>(y, x);
        if (!finiteDepth(current_depth, config_.router)) {
            continue;
        }
        const Eigen::Vector3f current_point(
            (static_cast<float>(x) - intrinsics_.cx) *
                current_depth / intrinsics_.fx,
            (static_cast<float>(y) - intrinsics_.cy) *
                current_depth / intrinsics_.fy,
            current_depth);
        const Eigen::Vector3f previous_prediction =
            parent_relative_pose.inverse() * current_point;
        cv::Point2f previous_pixel;
        if (!project(previous_prediction, intrinsics_, image_size,
                     previous_pixel)) {
            continue;
        }
        const int previous_x =
            static_cast<int>(std::lround(previous_pixel.x));
        const int previous_y =
            static_cast<int>(std::lround(previous_pixel.y));
        if (previous_static_mask.at<uchar>(previous_y, previous_x) == 0) {
            continue;
        }
        const float previous_depth =
            previous_depth_meters.at<float>(previous_y, previous_x);
        if (!finiteDepth(previous_depth, config_.router)) {
            continue;
        }
        const Eigen::Vector3f previous_point(
            (previous_pixel.x - intrinsics_.cx) *
                previous_depth / intrinsics_.fx,
            (previous_pixel.y - intrinsics_.cy) *
                previous_depth / intrinsics_.fy,
            previous_depth);
        if (!previous_point.allFinite()) {
            continue;
        }
        correspondences.push_back(
            Correspondence{previous_point, current_point});
    }
    result.geometric_support =
        static_cast<int>(correspondences.size());
    if (result.geometric_support <
        config_.router.min_support_points) {
        return result;
    }

    constexpr float kInlierThreshold = 0.05f;
    int initial_inliers = 0;
    result.initial_rmse = rmse(
        correspondences, parent_relative_pose,
        kInlierThreshold, &initial_inliers);
    Sophus::SE3f candidate = parent_relative_pose;
    Matrix6f last_information = Matrix6f::Zero();
    for (int iteration = 0; iteration < config_.max_iterations;
         ++iteration) {
        if (config_.translation_only) {
            Eigen::Matrix3f information = Eigen::Matrix3f::Zero();
            Eigen::Vector3f gradient = Eigen::Vector3f::Zero();
            for (const Correspondence& correspondence : correspondences) {
                const Eigen::Vector3f transformed =
                    candidate * correspondence.previous;
                const Eigen::Vector3f residual =
                    transformed - correspondence.current;
                const float residual_norm = residual.norm();
                const float weight =
                    huberWeight(residual_norm, config_.huber_delta);
                if (weight <= 0.0f) {
                    continue;
                }
                information.noalias() +=
                    weight * Eigen::Matrix3f::Identity();
                gradient.noalias() += weight * residual;
            }
            last_information.setZero();
            last_information.block<3, 3>(0, 0) = information;
            const Eigen::LDLT<Eigen::Matrix3f> decomposition(information);
            if (decomposition.info() != Eigen::Success) {
                break;
            }
            Eigen::Vector3f increment =
                -decomposition.solve(gradient);
            if (!increment.allFinite()) {
                break;
            }
            const Eigen::Vector3f accumulated =
                candidate.translation() -
                parent_relative_pose.translation();
            const Eigen::Vector3f proposed = accumulated + increment;
            if (config_.max_translation_correction > 0.0f &&
                proposed.norm() > config_.max_translation_correction) {
                increment =
                    proposed.normalized() *
                        config_.max_translation_correction -
                    accumulated;
            }
            candidate = Sophus::SE3f(
                parent_relative_pose.so3(),
                candidate.translation() + increment);
            if (increment.norm() < 1e-5f) {
                break;
            }
            continue;
        }

        Matrix6f information = Matrix6f::Zero();
        Eigen::Matrix<float, 6, 1> gradient =
            Eigen::Matrix<float, 6, 1>::Zero();
        for (const Correspondence& correspondence : correspondences) {
            const Eigen::Vector3f transformed =
                candidate * correspondence.previous;
            const Eigen::Vector3f residual =
                transformed - correspondence.current;
            const float residual_norm = residual.norm();
            const float weight =
                huberWeight(residual_norm, config_.huber_delta);
            if (weight <= 0.0f) {
                continue;
            }
            Eigen::Matrix<float, 3, 6> jacobian;
            jacobian.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
            jacobian.block<3, 3>(0, 3) = -skew(transformed);
            information.noalias() += weight *
                jacobian.transpose() * jacobian;
            gradient.noalias() += weight *
                jacobian.transpose() * residual;
        }
        last_information = 0.5f * (information + information.transpose());
        if (!last_information.allFinite()) {
            break;
        }
        const Eigen::LDLT<Matrix6f> decomposition(last_information);
        if (decomposition.info() != Eigen::Success) {
            break;
        }
        Eigen::Matrix<float, 6, 1> increment =
            -decomposition.solve(gradient);
        if (!increment.allFinite()) {
            break;
        }
        const float translation_norm = increment.head<3>().norm();
        const float rotation_norm = increment.tail<3>().norm();
        if (config_.max_translation_correction > 0.0f &&
            translation_norm > config_.max_translation_correction) {
            increment.head<3>() *=
                config_.max_translation_correction / translation_norm;
        }
        if (config_.max_rotation_correction > 0.0f &&
            rotation_norm > config_.max_rotation_correction) {
            increment.tail<3>() *=
                config_.max_rotation_correction / rotation_norm;
        }
        candidate = Sophus::SE3f::exp(increment) * candidate;
        if (increment.norm() < 1e-5f) {
            break;
        }
    }

    int final_inliers = 0;
    result.final_rmse = rmse(
        correspondences, candidate,
        kInlierThreshold, &final_inliers);
    result.inliers = final_inliers;
    result.information = last_information;
    result.refined_relative_pose = candidate;
    const Sophus::SE3f correction =
        candidate * parent_relative_pose.inverse();
    result.translation_correction = correction.translation().norm();
    result.rotation_correction = correction.so3().log().norm();
    result.improved = std::isfinite(result.initial_rmse) &&
        std::isfinite(result.final_rmse) &&
        result.final_rmse <
            result.initial_rmse *
                (1.0f - config_.min_rmse_improvement) &&
        final_inliers >= static_cast<int>(
            std::ceil(config_.min_inlier_ratio *
                      static_cast<float>(result.geometric_support)));
    result.valid = result.improved &&
        correction.matrix().allFinite() &&
        result.translation_correction <=
            config_.max_translation_correction + 1e-5f &&
        result.rotation_correction <=
            config_.max_rotation_correction + 1e-5f;
    return result;
}

GaussianBackgroundPoseRefiner::GaussianBackgroundPoseRefiner(
    const Config& config, const Intrinsics& intrinsics)
    : config_(config), intrinsics_(intrinsics)
{
    if (intrinsics_.fx <= 0.0f || intrinsics_.fy <= 0.0f ||
        config_.sample_stride <= 0 || config_.split_tile_size <= 0 ||
        config_.dynamic_dilation_radius < 0 ||
        config_.max_iterations <= 0 ||
        config_.min_proposal_support <= 0 ||
        config_.min_validation_support <= 0 ||
        config_.minimum_observable_rank < 1 ||
        config_.minimum_observable_rank > 3 ||
        config_.min_depth <= 0.0f ||
        config_.max_depth <= config_.min_depth ||
        config_.max_correspondence_distance <= 0.0f ||
        config_.huber_delta <= 0.0f ||
        config_.min_proposal_relative_improvement < 0.0f ||
        config_.min_proposal_relative_improvement >= 1.0f ||
        config_.min_validation_relative_improvement < 0.0f ||
        config_.min_validation_relative_improvement >= 1.0f ||
        config_.eigenvalue_relative_threshold <= 0.0f ||
        config_.eigenvalue_relative_threshold >= 1.0f ||
        config_.max_translation_correction <= 0.0f) {
        throw std::invalid_argument(
            "Invalid Gaussian background pose-refiner config");
    }
}

bool GaussianBackgroundPoseRefiner::finiteDepth(float depth) const
{
    return std::isfinite(depth) &&
        depth >= config_.min_depth && depth <= config_.max_depth;
}

Eigen::Vector3f GaussianBackgroundPoseRefiner::unproject(
    float x, float y, float depth) const
{
    return Eigen::Vector3f(
        (x - intrinsics_.cx) * depth / intrinsics_.fx,
        (y - intrinsics_.cy) * depth / intrinsics_.fy,
        depth);
}

bool GaussianBackgroundPoseRefiner::modelNormal(
    const cv::Mat& depth,
    int x,
    int y,
    Eigen::Vector3f& normal) const
{
    if (x <= 0 || y <= 0 || x >= depth.cols - 1 ||
        y >= depth.rows - 1) {
        return false;
    }
    const float left_depth = depth.at<float>(y, x - 1);
    const float right_depth = depth.at<float>(y, x + 1);
    const float up_depth = depth.at<float>(y - 1, x);
    const float down_depth = depth.at<float>(y + 1, x);
    if (!finiteDepth(left_depth) || !finiteDepth(right_depth) ||
        !finiteDepth(up_depth) || !finiteDepth(down_depth)) {
        return false;
    }
    const Eigen::Vector3f horizontal =
        unproject(static_cast<float>(x + 1), static_cast<float>(y),
                  right_depth) -
        unproject(static_cast<float>(x - 1), static_cast<float>(y),
                  left_depth);
    const Eigen::Vector3f vertical =
        unproject(static_cast<float>(x), static_cast<float>(y + 1),
                  down_depth) -
        unproject(static_cast<float>(x), static_cast<float>(y - 1),
                  up_depth);
    normal = horizontal.cross(vertical);
    const float norm = normal.norm();
    if (!normal.allFinite() || norm <= 1e-6f) {
        return false;
    }
    normal /= norm;
    if (normal.z() > 0.0f) {
        normal = -normal;
    }
    return true;
}

GaussianBackgroundPoseRefiner::Observation
GaussianBackgroundPoseRefiner::observe(
    const ModelPoint& model,
    const Eigen::Vector3f& translation,
    const cv::Mat& current_depth,
    const cv::Mat& safe_static_mask) const
{
    Observation observation;
    const Eigen::Vector3f transformed = model.point + translation;
    if (!transformed.allFinite() || !finiteDepth(transformed.z())) {
        return observation;
    }
    const float projected_x =
        intrinsics_.fx * transformed.x() / transformed.z() +
        intrinsics_.cx;
    const float projected_y =
        intrinsics_.fy * transformed.y() / transformed.z() +
        intrinsics_.cy;
    if (!std::isfinite(projected_x) || !std::isfinite(projected_y) ||
        projected_x < 0.0f || projected_y < 0.0f ||
        projected_x >= current_depth.cols ||
        projected_y >= current_depth.rows) {
        return observation;
    }
    const int x = static_cast<int>(std::lround(projected_x));
    const int y = static_cast<int>(std::lround(projected_y));
    if (x < 0 || y < 0 || x >= current_depth.cols ||
        y >= current_depth.rows ||
        safe_static_mask.at<uchar>(y, x) == 0) {
        return observation;
    }
    const float observed_depth = current_depth.at<float>(y, x);
    if (!finiteDepth(observed_depth)) {
        return observation;
    }
    const Eigen::Vector3f observed = unproject(
        projected_x, projected_y, observed_depth);
    const Eigen::Vector3f difference = transformed - observed;
    if (!difference.allFinite() ||
        difference.norm() > config_.max_correspondence_distance) {
        return observation;
    }
    observation.residual = model.normal.dot(difference);
    observation.valid = std::isfinite(observation.residual) &&
        std::abs(observation.residual) <=
            config_.max_correspondence_distance;
    return observation;
}

std::vector<GaussianBackgroundPoseRefiner::ModelPoint>
GaussianBackgroundPoseRefiner::collectModelPoints(
    const cv::Mat& gaussian_depth,
    const cv::Mat& safe_static_mask,
    int parity) const
{
    std::vector<ModelPoint> points;
    for (int y = config_.sample_stride;
         y < gaussian_depth.rows - config_.sample_stride;
         y += config_.sample_stride) {
        for (int x = config_.sample_stride;
             x < gaussian_depth.cols - config_.sample_stride;
             x += config_.sample_stride) {
            const int tile_parity =
                ((x / config_.split_tile_size) +
                 (y / config_.split_tile_size)) & 1;
            if (tile_parity != parity ||
                safe_static_mask.at<uchar>(y, x) == 0) {
                continue;
            }
            const float model_depth = gaussian_depth.at<float>(y, x);
            if (!finiteDepth(model_depth)) {
                continue;
            }
            Eigen::Vector3f normal;
            if (!modelNormal(gaussian_depth, x, y, normal)) {
                continue;
            }
            const Eigen::Vector3f point = unproject(
                static_cast<float>(x), static_cast<float>(y),
                model_depth);
            if (!point.allFinite()) {
                continue;
            }
            points.push_back(ModelPoint{cv::Point(x, y), point, normal});
        }
    }
    return points;
}

float GaussianBackgroundPoseRefiner::huberWeight(
    float residual, float delta)
{
    const float absolute = std::abs(residual);
    if (!std::isfinite(absolute)) {
        return 0.0f;
    }
    return absolute <= delta || absolute <= 1e-12f
        ? 1.0f
        : delta / absolute;
}

float GaussianBackgroundPoseRefiner::huberLoss(
    float residual, float delta)
{
    const float absolute = std::abs(residual);
    if (!std::isfinite(absolute)) {
        return std::numeric_limits<float>::infinity();
    }
    return absolute <= delta
        ? 0.5f * absolute * absolute
        : delta * (absolute - 0.5f * delta);
}

GaussianBackgroundPoseRefiner::Score
GaussianBackgroundPoseRefiner::scoreCommon(
    const std::vector<ModelPoint>& points,
    const Eigen::Vector3f& candidate_translation,
    const cv::Mat& current_depth,
    const cv::Mat& safe_static_mask,
    int minimum_support) const
{
    Score score;
    double parent_loss = 0.0;
    double candidate_loss = 0.0;
    for (const ModelPoint& point : points) {
        const Observation parent = observe(
            point, Eigen::Vector3f::Zero(), current_depth,
            safe_static_mask);
        const Observation candidate = observe(
            point, candidate_translation, current_depth,
            safe_static_mask);
        if (!parent.valid || !candidate.valid) {
            continue;
        }
        parent_loss += huberLoss(parent.residual, config_.huber_delta);
        candidate_loss +=
            huberLoss(candidate.residual, config_.huber_delta);
        ++score.common_support;
    }
    if (score.common_support == 0) {
        return score;
    }
    score.parent_loss = static_cast<float>(
        parent_loss / score.common_support);
    score.candidate_loss = static_cast<float>(
        candidate_loss / score.common_support);
    score.valid = score.common_support >= minimum_support &&
        std::isfinite(score.parent_loss) &&
        std::isfinite(score.candidate_loss);
    return score;
}

GaussianBackgroundPoseRefiner::Result
GaussianBackgroundPoseRefiner::refine(
    const cv::Mat& current_depth_meters,
    const cv::Mat& static_mask,
    const cv::Mat& dynamic_mask,
    const cv::Mat& gaussian_depth_at_parent,
    const Sophus::SE3f& parent_tcw) const
{
    Result result;
    result.refined_tcw = parent_tcw;
    if (!parent_tcw.matrix().allFinite() ||
        current_depth_meters.type() != CV_32FC1 ||
        gaussian_depth_at_parent.type() != CV_32FC1 ||
        current_depth_meters.size() != gaussian_depth_at_parent.size() ||
        static_mask.empty() || dynamic_mask.empty() ||
        static_mask.size() != current_depth_meters.size() ||
        dynamic_mask.size() != current_depth_meters.size() ||
        static_mask.channels() != 1 || dynamic_mask.channels() != 1) {
        return result;
    }

    cv::Mat static_binary;
    cv::Mat dynamic_binary;
    static_mask.convertTo(static_binary, CV_8UC1);
    dynamic_mask.convertTo(dynamic_binary, CV_8UC1);
    cv::threshold(static_binary, static_binary, 0, 255, cv::THRESH_BINARY);
    cv::threshold(dynamic_binary, dynamic_binary, 0, 255, cv::THRESH_BINARY);
    result.dynamic_pixels = cv::countNonZero(dynamic_binary);
    if (result.dynamic_pixels == 0) {
        return result;
    }
    if (config_.dynamic_dilation_radius > 0) {
        const int kernel_size = 2 * config_.dynamic_dilation_radius + 1;
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
        cv::dilate(dynamic_binary, dynamic_binary, kernel);
    }
    cv::Mat inverse_dynamic;
    cv::bitwise_not(dynamic_binary, inverse_dynamic);
    cv::Mat safe_static;
    cv::bitwise_and(static_binary, inverse_dynamic, safe_static);
    result.safe_static_pixels = cv::countNonZero(safe_static);

    const std::vector<ModelPoint> proposal = collectModelPoints(
        gaussian_depth_at_parent, safe_static, 0);
    const std::vector<ModelPoint> validation = collectModelPoints(
        gaussian_depth_at_parent, safe_static, 1);
    result.proposal_points = static_cast<int>(proposal.size());
    result.validation_points = static_cast<int>(validation.size());
    std::unordered_set<int> proposal_pixels;
    proposal_pixels.reserve(proposal.size());
    for (const ModelPoint& point : proposal) {
        proposal_pixels.insert(
            point.pixel.y * current_depth_meters.cols + point.pixel.x);
    }
    for (const ModelPoint& point : validation) {
        const int index =
            point.pixel.y * current_depth_meters.cols + point.pixel.x;
        if (proposal_pixels.count(index) != 0) {
            ++result.proposal_validation_overlap;
        }
    }
    if (result.proposal_points < config_.min_proposal_support ||
        result.validation_points < config_.min_validation_support ||
        result.proposal_validation_overlap != 0) {
        return result;
    }

    Eigen::Vector3f translation = Eigen::Vector3f::Zero();
    for (int iteration = 0; iteration < config_.max_iterations;
         ++iteration) {
        Eigen::Matrix3f information = Eigen::Matrix3f::Zero();
        Eigen::Vector3f gradient = Eigen::Vector3f::Zero();
        int support = 0;
        for (const ModelPoint& point : proposal) {
            const Observation observation = observe(
                point, translation, current_depth_meters, safe_static);
            if (!observation.valid) {
                continue;
            }
            const float weight = huberWeight(
                observation.residual, config_.huber_delta);
            if (weight <= 0.0f) {
                continue;
            }
            information.noalias() +=
                weight * point.normal * point.normal.transpose();
            gradient.noalias() +=
                weight * point.normal * observation.residual;
            ++support;
        }
        result.proposal_support = support;
        if (support < config_.min_proposal_support ||
            !information.allFinite() || !gradient.allFinite()) {
            return result;
        }
        information = 0.5f * (information + information.transpose());
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(
            information);
        if (eigensolver.info() != Eigen::Success) {
            return result;
        }
        const Eigen::Vector3f eigenvalues = eigensolver.eigenvalues();
        const float maximum_eigenvalue = eigenvalues.maxCoeff();
        if (!std::isfinite(maximum_eigenvalue) ||
            maximum_eigenvalue <= 1e-8f) {
            return result;
        }
        Eigen::Vector3f inverse_eigenvalues = Eigen::Vector3f::Zero();
        result.observable_rank = 0;
        for (int index = 0; index < 3; ++index) {
            if (eigenvalues[index] >
                maximum_eigenvalue *
                    config_.eigenvalue_relative_threshold) {
                inverse_eigenvalues[index] = 1.0f / eigenvalues[index];
                ++result.observable_rank;
            }
        }
        if (result.observable_rank < config_.minimum_observable_rank) {
            return result;
        }
        const Eigen::Vector3f increment =
            -eigensolver.eigenvectors() *
            inverse_eigenvalues.asDiagonal() *
            eigensolver.eigenvectors().transpose() * gradient;
        if (!increment.allFinite()) {
            return result;
        }
        translation += increment;
        if (translation.norm() > config_.max_translation_correction) {
            translation = translation.normalized() *
                config_.max_translation_correction;
        }
        if (increment.norm() < 1e-5f) {
            break;
        }
    }

    result.translation_correction = translation.norm();
    if (!translation.allFinite() ||
        result.translation_correction <= 1e-6f ||
        result.translation_correction >
            config_.max_translation_correction + 1e-6f) {
        return result;
    }
    result.refined_tcw = Sophus::SE3f(
        parent_tcw.so3(), parent_tcw.translation() + translation);
    const Sophus::SE3f correction =
        result.refined_tcw * parent_tcw.inverse();
    result.translation_correction = correction.translation().norm();
    result.rotation_correction = correction.so3().log().norm();
    result.candidate_available = result.refined_tcw.matrix().allFinite() &&
        result.translation_correction <=
            config_.max_translation_correction + 1e-6f &&
        result.rotation_correction <= 1e-6f;
    if (!result.candidate_available) {
        return result;
    }

    const Score proposal_score = scoreCommon(
        proposal, translation, current_depth_meters, safe_static,
        config_.min_proposal_support);
    result.proposal_parent_loss = proposal_score.parent_loss;
    result.proposal_candidate_loss = proposal_score.candidate_loss;
    result.proposal_improved = proposal_score.valid &&
        proposal_score.candidate_loss <
            proposal_score.parent_loss *
                (1.0f - config_.min_proposal_relative_improvement);

    const Score validation_score = scoreCommon(
        validation, translation, current_depth_meters, safe_static,
        config_.min_validation_support);
    result.validation_common_support = validation_score.common_support;
    result.validation_parent_loss = validation_score.parent_loss;
    result.validation_candidate_loss = validation_score.candidate_loss;
    result.validation_valid = validation_score.valid;
    result.validation_prefers_candidate = validation_score.valid &&
        validation_score.candidate_loss <
            validation_score.parent_loss *
                (1.0f - config_.min_validation_relative_improvement);
    result.gate_pass = result.candidate_available &&
        result.proposal_improved && result.validation_valid &&
        result.validation_prefers_candidate &&
        result.proposal_validation_overlap == 0;
    return result;
}

}  // namespace Motion3D
