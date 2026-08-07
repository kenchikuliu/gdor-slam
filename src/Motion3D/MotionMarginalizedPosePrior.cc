#include "Motion3D/MotionMarginalizedPosePrior.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>

namespace Motion3D {
namespace {

float intersectionOverUnion(const cv::Rect& lhs, const cv::Rect& rhs)
{
    const cv::Rect intersection = lhs & rhs;
    const float intersection_area = static_cast<float>(intersection.area());
    const float union_area = static_cast<float>(lhs.area() + rhs.area() - intersection.area());
    return union_area > 0.0f ? intersection_area / union_area : 0.0f;
}

Eigen::Matrix3f skew(const Eigen::Vector3f& value)
{
    Eigen::Matrix3f result;
    result << 0.0f, -value.z(), value.y(),
              value.z(), 0.0f, -value.x(),
              -value.y(), value.x(), 0.0f;
    return result;
}

bool fitRigid(const std::vector<Eigen::Vector3f>& source,
              const std::vector<Eigen::Vector3f>& target,
              const std::vector<int>& indices,
              Sophus::SE3f& transform)
{
    if (indices.size() < 3) return false;

    Eigen::Vector3f source_mean = Eigen::Vector3f::Zero();
    Eigen::Vector3f target_mean = Eigen::Vector3f::Zero();
    for (int index : indices) {
        source_mean += source[index];
        target_mean += target[index];
    }
    source_mean /= static_cast<float>(indices.size());
    target_mean /= static_cast<float>(indices.size());

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (int index : indices) {
        covariance += (target[index] - target_mean) *
                      (source[index] - source_mean).transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(
        covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.singularValues()(1) < 1e-7f) return false;

    Eigen::Matrix3f correction = Eigen::Matrix3f::Identity();
    if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.0f) {
        correction(2, 2) = -1.0f;
    }
    const Eigen::Matrix3f rotation =
        svd.matrixU() * correction * svd.matrixV().transpose();
    const Eigen::Vector3f translation = target_mean - rotation * source_mean;
    transform = Sophus::SE3f(rotation, translation);
    return transform.matrix().allFinite();
}

cv::Mat normalizeDynamicMask(const cv::Mat& mask, const cv::Size& size)
{
    if (mask.empty() || mask.size() != size) return cv::Mat();
    cv::Mat binary;
    if (mask.type() != CV_8UC1) {
        mask.convertTo(binary, CV_8UC1);
    } else {
        binary = mask.clone();
    }
    cv::threshold(binary, binary, 0, 255, cv::THRESH_BINARY);
    return binary;
}

template <typename Derived>
typename Derived::PlainObject symmetrized(
    const Eigen::MatrixBase<Derived>& matrix)
{
    return 0.5f * (matrix + matrix.transpose());
}

bool invertPositiveDefinite(
    const MotionMarginalizedPosePrior::Matrix6f& matrix,
    MotionMarginalizedPosePrior::Matrix6f& inverse)
{
    Eigen::LDLT<MotionMarginalizedPosePrior::Matrix6f> decomposition(
        symmetrized(matrix));
    if (decomposition.info() != Eigen::Success ||
        (decomposition.vectorD().array() <= 1e-9f).any()) {
        return false;
    }
    inverse = decomposition.solve(MotionMarginalizedPosePrior::Matrix6f::Identity());
    return decomposition.info() == Eigen::Success && inverse.allFinite();
}

}  // namespace

MotionMarginalizedPosePrior::MotionMarginalizedPosePrior(
    const Config& config, const CameraIntrinsics& intrinsics)
    : config_(config), intrinsics_(intrinsics), rng_(config.seed)
{
    config_.max_features = std::max(1, config_.max_features);
    config_.min_features_per_object = std::max(3, config_.min_features_per_object);
    config_.min_component_area = std::max(1, config_.min_component_area);
    config_.min_track_age = std::max(1, config_.min_track_age);
    config_.ransac_iterations = std::max(1, config_.ransac_iterations);
    config_.ransac_threshold = std::max(1e-4f, config_.ransac_threshold);
    config_.min_inlier_ratio = std::clamp(config_.min_inlier_ratio, 0.0f, 1.0f);
    config_.min_information = std::max(0.0f, config_.min_information);
    config_.foreground_depth_separation =
        std::max(0.0f, config_.foreground_depth_separation);
    config_.foreground_min_fraction =
        std::clamp(config_.foreground_min_fraction, 0.01f, 0.49f);
    config_.min_object_translation_speed =
        std::max(0.0f, config_.min_object_translation_speed);
    config_.min_object_rotation_speed =
        std::max(0.0f, config_.min_object_rotation_speed);
    config_.measurement_translation_sigma =
        std::max(1e-4f, config_.measurement_translation_sigma);
    config_.measurement_rotation_sigma =
        std::max(1e-4f, config_.measurement_rotation_sigma);
    config_.camera_translation_sigma =
        std::max(1e-4f, config_.camera_translation_sigma);
    config_.camera_rotation_sigma =
        std::max(1e-4f, config_.camera_rotation_sigma);
    config_.velocity_process_translation_sigma =
        std::max(0.0f, config_.velocity_process_translation_sigma);
    config_.velocity_process_rotation_sigma =
        std::max(0.0f, config_.velocity_process_rotation_sigma);
    config_.max_information_eigenvalue =
        std::max(config_.min_information, config_.max_information_eigenvalue);
    config_.candidate_mahalanobis_threshold =
        std::max(0.0f, config_.candidate_mahalanobis_threshold);
}

cv::Mat MotionMarginalizedPosePrior::refineForegroundMask(
    const cv::Mat& semantic_mask, const cv::Mat& depth) const
{
    if (!config_.use_depth_foreground_filter || semantic_mask.empty() ||
        depth.empty() || semantic_mask.size() != depth.size()) {
        return semantic_mask.clone();
    }

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(
        semantic_mask, labels, stats, centroids, 8, CV_32S);
    cv::Mat foreground = cv::Mat::zeros(semantic_mask.size(), CV_8UC1);

    for (int label = 1; label < count; ++label) {
        std::vector<float> values;
        values.reserve(stats.at<int>(label, cv::CC_STAT_AREA));
        for (int y = 0; y < labels.rows; ++y) {
            const int* label_row = labels.ptr<int>(y);
            const float* depth_row = depth.ptr<float>(y);
            for (int x = 0; x < labels.cols; ++x) {
                if (label_row[x] == label && std::isfinite(depth_row[x]) &&
                    depth_row[x] > 0.0f) {
                    values.push_back(depth_row[x]);
                }
            }
        }
        if (values.size() < static_cast<std::size_t>(config_.min_features_per_object)) {
            foreground.setTo(255, labels == label);
            continue;
        }

        std::sort(values.begin(), values.end());
        float near_center = values[values.size() / 4];
        float far_center = values[(3 * values.size()) / 4];
        for (int iteration = 0; iteration < 8; ++iteration) {
            float near_sum = 0.0f;
            float far_sum = 0.0f;
            int near_count = 0;
            int far_count = 0;
            for (float value : values) {
                if (std::abs(value - near_center) <= std::abs(value - far_center)) {
                    near_sum += value;
                    ++near_count;
                } else {
                    far_sum += value;
                    ++far_count;
                }
            }
            if (near_count == 0 || far_count == 0) break;
            near_center = near_sum / static_cast<float>(near_count);
            far_center = far_sum / static_cast<float>(far_count);
            if (near_center > far_center) std::swap(near_center, far_center);
        }

        const int near_count = static_cast<int>(std::count_if(
            values.begin(), values.end(), [near_center, far_center](float value) {
                return std::abs(value - near_center) <= std::abs(value - far_center);
            }));
        const float near_fraction =
            static_cast<float>(near_count) / static_cast<float>(values.size());
        const bool separated =
            far_center - near_center >= config_.foreground_depth_separation &&
            near_fraction >= config_.foreground_min_fraction &&
            near_fraction <= 1.0f - config_.foreground_min_fraction;
        if (!separated) {
            foreground.setTo(255, labels == label);
            continue;
        }

        for (int y = 0; y < labels.rows; ++y) {
            const int* label_row = labels.ptr<int>(y);
            const float* depth_row = depth.ptr<float>(y);
            unsigned char* foreground_row = foreground.ptr<unsigned char>(y);
            for (int x = 0; x < labels.cols; ++x) {
                if (label_row[x] != label || !std::isfinite(depth_row[x]) ||
                    depth_row[x] <= 0.0f) {
                    continue;
                }
                if (std::abs(depth_row[x] - near_center) <=
                    std::abs(depth_row[x] - far_center)) {
                    foreground_row[x] = 255;
                }
            }
        }
    }
    return foreground;
}

std::vector<MotionMarginalizedPosePrior::Component>
MotionMarginalizedPosePrior::extractComponents(const cv::Mat& dynamic_mask,
                                               cv::Mat& labels) const
{
    std::vector<Component> components;
    if (dynamic_mask.empty()) return components;

    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(
        dynamic_mask, labels, stats, centroids, 8, CV_32S);
    for (int label = 1; label < count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < config_.min_component_area) continue;
        components.push_back(Component{
            label,
            cv::Rect(stats.at<int>(label, cv::CC_STAT_LEFT),
                     stats.at<int>(label, cv::CC_STAT_TOP),
                     stats.at<int>(label, cv::CC_STAT_WIDTH),
                     stats.at<int>(label, cv::CC_STAT_HEIGHT)),
            area});
    }
    return components;
}

std::vector<int> MotionMarginalizedPosePrior::matchStatesToComponents(
    const std::vector<Component>& components) const
{
    struct Pair {
        float iou;
        int component;
        int state;
    };
    std::vector<Pair> pairs;
    for (std::size_t component = 0; component < components.size(); ++component) {
        for (std::size_t state = 0; state < object_states_.size(); ++state) {
            const float iou = intersectionOverUnion(
                components[component].bbox, object_states_[state].bbox);
            if (iou > 0.1f) {
                pairs.push_back(Pair{
                    iou, static_cast<int>(component), static_cast<int>(state)});
            }
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& lhs, const Pair& rhs) {
        return lhs.iou > rhs.iou;
    });
    std::vector<int> assignment(components.size(), -1);
    std::vector<bool> state_used(object_states_.size(), false);
    for (const Pair& pair : pairs) {
        if (assignment[pair.component] < 0 && !state_used[pair.state]) {
            assignment[pair.component] = pair.state;
            state_used[pair.state] = true;
        }
    }
    return assignment;
}

Eigen::Vector3f MotionMarginalizedPosePrior::backProject(
    const cv::Point2f& pixel, float depth) const
{
    return Eigen::Vector3f(
        (pixel.x - intrinsics_.cx) * depth / intrinsics_.fx,
        (pixel.y - intrinsics_.cy) * depth / intrinsics_.fy,
        depth);
}

bool MotionMarginalizedPosePrior::estimateRigidTransform(
    const std::vector<Eigen::Vector3f>& source,
    const std::vector<Eigen::Vector3f>& target,
    float inlier_threshold,
    int ransac_iterations,
    std::mt19937& rng,
    Sophus::SE3f& transform,
    std::vector<unsigned char>& inliers,
    float& rmse)
{
    if (source.size() != target.size() || source.size() < 3) return false;

    const int count = static_cast<int>(source.size());
    std::uniform_int_distribution<int> sample(0, count - 1);
    int best_count = 0;
    float best_error = std::numeric_limits<float>::infinity();
    Sophus::SE3f best_transform;
    std::vector<unsigned char> best_inliers(source.size(), 0);

    for (int iteration = 0; iteration < ransac_iterations; ++iteration) {
        std::vector<int> indices;
        indices.reserve(3);
        while (indices.size() < 3) {
            const int index = sample(rng);
            if (std::find(indices.begin(), indices.end(), index) == indices.end()) {
                indices.push_back(index);
            }
        }

        Sophus::SE3f candidate;
        if (!fitRigid(source, target, indices, candidate)) continue;

        int candidate_count = 0;
        float squared_error = 0.0f;
        std::vector<unsigned char> candidate_inliers(source.size(), 0);
        for (int i = 0; i < count; ++i) {
            const float error = (candidate * source[i] - target[i]).norm();
            if (error <= inlier_threshold) {
                candidate_inliers[i] = 1;
                ++candidate_count;
                squared_error += error * error;
            }
        }

        if (candidate_count > best_count ||
            (candidate_count == best_count && squared_error < best_error)) {
            best_count = candidate_count;
            best_error = squared_error;
            best_transform = candidate;
            best_inliers.swap(candidate_inliers);
        }
    }

    if (best_count < 3) return false;
    std::vector<int> all_inliers;
    all_inliers.reserve(best_count);
    for (int i = 0; i < count; ++i) {
        if (best_inliers[i]) all_inliers.push_back(i);
    }
    if (!fitRigid(source, target, all_inliers, transform)) return false;

    float squared_error = 0.0f;
    int refined_count = 0;
    inliers.assign(source.size(), 0);
    for (int i = 0; i < count; ++i) {
        const float error = (transform * source[i] - target[i]).norm();
        if (error <= inlier_threshold) {
            inliers[i] = 1;
            squared_error += error * error;
            ++refined_count;
        }
    }
    if (refined_count < 3) return false;
    rmse = std::sqrt(squared_error / static_cast<float>(refined_count));
    return true;
}

MotionMarginalizedPosePrior::SchurResult
MotionMarginalizedPosePrior::linearizeAndMarginalize(
    const std::vector<Eigen::Vector3f>& source,
    const std::vector<Eigen::Vector3f>& target,
    const std::vector<unsigned char>& inliers,
    const Sophus::SE3f& camera_motion,
    const Sophus::SE3f& object_motion,
    float residual_variance,
    const Matrix6f& object_prior_information)
{
    SchurResult result;
    if (source.size() != target.size() || source.size() != inliers.size() ||
        source.size() < 3 || !camera_motion.matrix().allFinite() ||
        !object_motion.matrix().allFinite() ||
        !object_prior_information.allFinite()) {
        return result;
    }

    const float inverse_variance = 1.0f / std::max(1e-9f, residual_variance);
    const Eigen::Matrix3f camera_rotation = camera_motion.so3().matrix();
    int inlier_count = 0;
    for (std::size_t i = 0; i < source.size(); ++i) {
        if (!inliers[i]) continue;

        const Eigen::Vector3f object_point = object_motion * source[i];
        const Eigen::Vector3f camera_point = camera_motion * object_point;
        const Eigen::Vector3f residual = camera_point - target[i];

        Eigen::Matrix<float, 3, 6> camera_jacobian;
        camera_jacobian.template block<3, 3>(0, 0) =
            Eigen::Matrix3f::Identity();
        camera_jacobian.template block<3, 3>(0, 3) =
            -skew(camera_point);

        Eigen::Matrix<float, 3, 6> object_jacobian;
        object_jacobian.template block<3, 3>(0, 0) = camera_rotation;
        object_jacobian.template block<3, 3>(0, 3) =
            -camera_rotation * skew(object_point);

        result.Hcc.noalias() += inverse_variance *
            camera_jacobian.transpose() * camera_jacobian;
        result.Hco.noalias() += inverse_variance *
            camera_jacobian.transpose() * object_jacobian;
        result.Hoo.noalias() += inverse_variance *
            object_jacobian.transpose() * object_jacobian;
        result.bc.noalias() += inverse_variance *
            camera_jacobian.transpose() * residual;
        result.bo.noalias() += inverse_variance *
            object_jacobian.transpose() * residual;
        ++inlier_count;
    }
    if (inlier_count < 3) return result;

    Matrix6f inverse_object_block;
    if (!invertPositiveDefinite(
            result.Hoo + symmetrized(object_prior_information),
            inverse_object_block)) {
        return result;
    }
    result.camera_information = symmetrized(
        result.Hcc - result.Hco * inverse_object_block *
                         result.Hco.transpose());
    result.camera_gradient =
        result.bc - result.Hco * inverse_object_block * result.bo;

    Matrix6f inverse_camera_information;
    if (!invertPositiveDefinite(
            result.camera_information, inverse_camera_information)) {
        return result;
    }
    result.camera_increment =
        -inverse_camera_information * result.camera_gradient;
    result.valid = result.camera_information.allFinite() &&
                   result.camera_gradient.allFinite() &&
                   result.camera_increment.allFinite();
    return result;
}

float MotionMarginalizedPosePrior::minimumInformationEigenvalue(
    const Matrix6f& information)
{
    Eigen::SelfAdjointEigenSolver<Matrix6f> solver(symmetrized(information));
    if (solver.info() != Eigen::Success) return 0.0f;
    return std::max(0.0f, solver.eigenvalues().minCoeff());
}

MotionMarginalizedPosePrior::Matrix6f
MotionMarginalizedPosePrior::transformCovariance(
    const Matrix6f& covariance,
    const Sophus::SE3f& target_from_source)
{
    const Matrix6f adjoint = target_from_source.Adj();
    return symmetrized(adjoint * covariance * adjoint.transpose());
}

MotionMarginalizedPosePrior::Matrix6f
MotionMarginalizedPosePrior::leftExpJacobian(const Vector6f& tangent)
{
    constexpr float epsilon = 1e-4f;
    const Sophus::SE3f base = Sophus::SE3f::exp(tangent);
    const Sophus::SE3f base_inverse = base.inverse();
    Matrix6f jacobian;
    for (int axis = 0; axis < 6; ++axis) {
        Vector6f perturbation = Vector6f::Zero();
        perturbation[axis] = epsilon;
        const Vector6f plus =
            (Sophus::SE3f::exp(tangent + perturbation) * base_inverse).log();
        const Vector6f minus =
            (Sophus::SE3f::exp(tangent - perturbation) * base_inverse).log();
        jacobian.col(axis) = (plus - minus) / (2.0f * epsilon);
    }
    return jacobian;
}

MotionMarginalizedPosePrior::Matrix6f
MotionMarginalizedPosePrior::leftLogJacobian(
    const Sophus::SE3f& transform)
{
    constexpr float epsilon = 1e-4f;
    Matrix6f jacobian;
    for (int axis = 0; axis < 6; ++axis) {
        Vector6f perturbation = Vector6f::Zero();
        perturbation[axis] = epsilon;
        const Vector6f plus =
            (Sophus::SE3f::exp(perturbation) * transform).log();
        const Vector6f minus =
            (Sophus::SE3f::exp(-perturbation) * transform).log();
        jacobian.col(axis) = (plus - minus) / (2.0f * epsilon);
    }
    return jacobian;
}

float MotionMarginalizedPosePrior::candidateInnovationMahalanobis(
    const CameraCandidate& lhs,
    const CameraCandidate& rhs)
{
    Matrix6f lhs_covariance;
    Matrix6f rhs_covariance;
    if (!invertPositiveDefinite(lhs.information, lhs_covariance) ||
        !invertPositiveDefinite(rhs.information, rhs_covariance)) {
        return std::numeric_limits<float>::infinity();
    }
    Matrix6f innovation_information;
    if (!invertPositiveDefinite(
            lhs_covariance + rhs_covariance,
            innovation_information)) {
        return std::numeric_limits<float>::infinity();
    }
    const Vector6f innovation = (lhs.pose * rhs.pose.inverse()).log();
    const float squared_distance =
        innovation.dot(innovation_information * innovation);
    return std::isfinite(squared_distance)
        ? std::max(0.0f, squared_distance)
        : std::numeric_limits<float>::infinity();
}

std::vector<std::size_t>
MotionMarginalizedPosePrior::selectConsistentCandidates(
    const std::vector<CameraCandidate>& candidates,
    float mahalanobis_threshold)
{
    std::vector<std::size_t> best;
    float best_information_score = -1.0f;
    for (std::size_t seed = 0; seed < candidates.size(); ++seed) {
        std::vector<std::size_t> order;
        order.reserve(candidates.size() - 1);
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            if (index != seed) order.push_back(index);
        }
        std::sort(
            order.begin(), order.end(),
            [&candidates](std::size_t lhs, std::size_t rhs) {
                return minimumInformationEigenvalue(
                           candidates[lhs].information) >
                       minimumInformationEigenvalue(
                           candidates[rhs].information);
            });

        std::vector<std::size_t> consensus{seed};
        for (std::size_t index : order) {
            const bool consistent = std::all_of(
                consensus.begin(), consensus.end(),
                [&candidates, index, mahalanobis_threshold](
                    std::size_t accepted) {
                    const float forward = candidateInnovationMahalanobis(
                        candidates[index], candidates[accepted]);
                    const float reverse = candidateInnovationMahalanobis(
                        candidates[accepted], candidates[index]);
                    return std::max(forward, reverse) <=
                           mahalanobis_threshold;
                });
            if (consistent) consensus.push_back(index);
        }

        float information_score = 0.0f;
        for (std::size_t index : consensus) {
            information_score += minimumInformationEigenvalue(
                candidates[index].information);
        }
        if (consensus.size() > best.size() ||
            (consensus.size() == best.size() &&
             information_score > best_information_score)) {
            best = std::move(consensus);
            best_information_score = information_score;
        }
    }
    std::sort(best.begin(), best.end());
    return best;
}

MotionMarginalizedPosePrior::Matrix6f
MotionMarginalizedPosePrior::diagonalCovariance(
    float translation_sigma, float rotation_sigma) const
{
    Matrix6f covariance = Matrix6f::Zero();
    covariance.diagonal().template head<3>().setConstant(
        translation_sigma * translation_sigma);
    covariance.diagonal().template tail<3>().setConstant(
        rotation_sigma * rotation_sigma);
    return covariance;
}

MotionMarginalizedPosePrior::Matrix6f
MotionMarginalizedPosePrior::clampInformation(
    const Matrix6f& information) const
{
    Eigen::SelfAdjointEigenSolver<Matrix6f> solver(symmetrized(information));
    if (solver.info() != Eigen::Success) return Matrix6f::Zero();
    Eigen::Matrix<float, 6, 1> eigenvalues = solver.eigenvalues();
    for (int i = 0; i < eigenvalues.size(); ++i) {
        eigenvalues[i] = std::clamp(
            eigenvalues[i], 0.0f, config_.max_information_eigenvalue);
    }
    return symmetrized(
        solver.eigenvectors() * eigenvalues.asDiagonal() *
        solver.eigenvectors().transpose());
}

std::vector<int> MotionMarginalizedPosePrior::activeObjectIds() const
{
    std::vector<int> ids;
    ids.reserve(object_states_.size());
    for (const ObjectState& state : object_states_) ids.push_back(state.id);
    return ids;
}

MotionMarginalizedPosePrior::Result MotionMarginalizedPosePrior::prepare(
    const cv::Mat& rgb,
    const cv::Mat& depth_meters,
    const cv::Mat& semantic_dynamic_mask,
    double timestamp)
{
    Result result;
    pending_estimates_.clear();
    current_components_.clear();
    pending_dt_ = previous_gray_.empty() ? 0.0 : timestamp - previous_timestamp_;

    if (rgb.empty() || depth_meters.empty() || depth_meters.type() != CV_32FC1 ||
        intrinsics_.fx <= 0.0f || intrinsics_.fy <= 0.0f) {
        return result;
    }

    cv::Mat gray;
    if (rgb.channels() == 3) {
        cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
    } else if (rgb.channels() == 4) {
        cv::cvtColor(rgb, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = rgb.clone();
    }
    const cv::Mat semantic_mask =
        normalizeDynamicMask(semantic_dynamic_mask, gray.size());
    if (semantic_mask.empty() || depth_meters.size() != gray.size()) {
        cacheCurrentFrame(gray, depth_meters, cv::Mat::zeros(gray.size(), CV_8UC1), timestamp);
        return result;
    }
    const cv::Mat current_mask = refineForegroundMask(semantic_mask, depth_meters);
    const int semantic_pixels = cv::countNonZero(semantic_mask);
    result.foreground_fraction = semantic_pixels > 0
        ? static_cast<float>(cv::countNonZero(current_mask)) /
              static_cast<float>(semantic_pixels)
        : 0.0f;

    cv::Mat current_labels;
    current_components_ = extractComponents(current_mask, current_labels);
    if (previous_gray_.empty() || previous_depth_.empty() ||
        previous_dynamic_mask_.empty() || !previous_pose_valid_ ||
        pending_dt_ <= 1e-6) {
        object_states_.clear();
        for (const Component& component : current_components_) {
            object_states_.push_back(ObjectState{next_object_id_++, component.bbox});
        }
        cacheCurrentFrame(gray, depth_meters, current_mask, timestamp);
        return result;
    }

    cv::Mat previous_labels;
    const std::vector<Component> previous_components =
        extractComponents(previous_dynamic_mask_, previous_labels);
    const std::vector<int> previous_assignments =
        matchStatesToComponents(previous_components);
    const int per_object_limit = std::max(
        config_.min_features_per_object,
        config_.max_features / std::max(1, static_cast<int>(previous_components.size())));

    struct Correspondences {
        int previous_object_id = -1;
        int current_label = 0;
        cv::Rect current_bbox;
        std::vector<Eigen::Vector3f> source;
        std::vector<Eigen::Vector3f> target;
    };
    std::map<std::pair<int, int>, Correspondences> grouped;

    for (std::size_t component_index = 0;
         component_index < previous_components.size(); ++component_index) {
        const Component& component = previous_components[component_index];
        const int state_index = previous_assignments[component_index];
        if (state_index < 0) continue;

        cv::Mat feature_mask = previous_labels == component.label;
        std::vector<cv::Point2f> previous_points;
        cv::goodFeaturesToTrack(previous_gray_, previous_points, per_object_limit,
                                0.01, 5.0, feature_mask);
        if (previous_points.empty()) continue;

        std::vector<cv::Point2f> current_points;
        std::vector<unsigned char> forward_status;
        std::vector<float> forward_error;
        cv::calcOpticalFlowPyrLK(previous_gray_, gray, previous_points,
                                 current_points, forward_status, forward_error);
        std::vector<cv::Point2f> backward_points;
        std::vector<unsigned char> backward_status;
        std::vector<float> backward_error;
        cv::calcOpticalFlowPyrLK(gray, previous_gray_, current_points,
                                 backward_points, backward_status, backward_error);

        for (std::size_t i = 0; i < previous_points.size(); ++i) {
            if (!forward_status[i] || !backward_status[i] ||
                cv::norm(previous_points[i] - backward_points[i]) >
                    config_.lk_forward_backward_threshold) {
                continue;
            }
            const int px = cvRound(previous_points[i].x);
            const int py = cvRound(previous_points[i].y);
            const int cx = cvRound(current_points[i].x);
            const int cy = cvRound(current_points[i].y);
            if (px < 0 || py < 0 || px >= previous_depth_.cols || py >= previous_depth_.rows ||
                cx < 0 || cy < 0 || cx >= depth_meters.cols || cy >= depth_meters.rows) {
                continue;
            }
            const int current_label = current_labels.at<int>(cy, cx);
            if (current_label <= 0 || current_mask.at<unsigned char>(cy, cx) == 0) continue;
            auto component_it = std::find_if(
                current_components_.begin(), current_components_.end(),
                [current_label](const Component& item) { return item.label == current_label; });
            if (component_it == current_components_.end()) continue;

            const float previous_depth = previous_depth_.at<float>(py, px);
            const float current_depth = depth_meters.at<float>(cy, cx);
            if (!std::isfinite(previous_depth) || !std::isfinite(current_depth) ||
                previous_depth <= 0.0f || current_depth <= 0.0f) {
                continue;
            }

            const auto key = std::make_pair(object_states_[state_index].id, current_label);
            Correspondences& correspondences = grouped[key];
            correspondences.previous_object_id = object_states_[state_index].id;
            correspondences.current_label = current_label;
            correspondences.current_bbox = component_it->bbox;
            correspondences.source.push_back(backProject(previous_points[i], previous_depth));
            correspondences.target.push_back(backProject(current_points[i], current_depth));
        }
    }

    struct EvaluatedEstimate {
        std::size_t pending_index = 0;
        const Correspondences* correspondences = nullptr;
        std::vector<unsigned char> inliers;
        float residual_variance = 0.0f;
    };
    std::vector<EvaluatedEstimate> evaluated_estimates;
    for (const auto& entry : grouped) {
        const Correspondences& correspondences = entry.second;
        result.feature_tracks += static_cast<int>(correspondences.source.size());
        if (correspondences.source.size() <
            static_cast<std::size_t>(config_.min_features_per_object)) {
            continue;
        }

        Sophus::SE3f apparent;
        std::vector<unsigned char> inliers;
        float rmse = 0.0f;
        if (!estimateRigidTransform(correspondences.source, correspondences.target,
                                    config_.ransac_threshold, config_.ransac_iterations,
                                    rng_, apparent, inliers, rmse)) {
            continue;
        }
        const int inlier_count = std::accumulate(inliers.begin(), inliers.end(), 0);
        const float inlier_ratio = static_cast<float>(inlier_count) /
                                   static_cast<float>(inliers.size());
        if (inlier_ratio < config_.min_inlier_ratio) continue;

        const float residual_variance = std::max(
            rmse * rmse,
            config_.ransac_threshold * config_.ransac_threshold * 0.04f);
        const SchurResult apparent_linearization = linearizeAndMarginalize(
            correspondences.source, correspondences.target, inliers,
            apparent, Sophus::SE3f(), residual_variance,
            Matrix6f::Identity());
        ApparentEstimate estimate;
        estimate.previous_object_id = correspondences.previous_object_id;
        estimate.current_label = correspondences.current_label;
        estimate.current_bbox = correspondences.current_bbox;
        estimate.apparent_transform = apparent;
        estimate.track_count = static_cast<int>(correspondences.source.size());
        estimate.inlier_count = inlier_count;
        estimate.inlier_ratio = inlier_ratio;
        estimate.rmse = rmse;
        estimate.observation_information = apparent_linearization.Hcc;
        pending_estimates_.push_back(estimate);
        evaluated_estimates.push_back(EvaluatedEstimate{
            pending_estimates_.size() - 1, &correspondences, inliers,
            residual_variance});
        ++result.objects_observed;
        result.inlier_tracks += inlier_count;
    }

    std::sort(
        evaluated_estimates.begin(), evaluated_estimates.end(),
        [this](const EvaluatedEstimate& lhs, const EvaluatedEstimate& rhs) {
            const ApparentEstimate& lhs_estimate =
                pending_estimates_[lhs.pending_index];
            const ApparentEstimate& rhs_estimate =
                pending_estimates_[rhs.pending_index];
            if (lhs_estimate.inlier_count != rhs_estimate.inlier_count) {
                return lhs_estimate.inlier_count > rhs_estimate.inlier_count;
            }
            return lhs_estimate.inlier_ratio > rhs_estimate.inlier_ratio;
        });

    std::vector<CameraCandidate> camera_candidates;
    std::set<int> used_previous_ids;
    std::set<int> used_current_labels;
    const float dt = static_cast<float>(pending_dt_);
    const Matrix6f velocity_process_covariance = diagonalCovariance(
        config_.velocity_process_translation_sigma,
        config_.velocity_process_rotation_sigma);
    for (const EvaluatedEstimate& evaluated : evaluated_estimates) {
        const ApparentEstimate& estimate =
            pending_estimates_[evaluated.pending_index];
        if (!used_previous_ids.insert(estimate.previous_object_id).second ||
            !used_current_labels.insert(estimate.current_label).second) {
            continue;
        }

        auto state_it = std::find_if(
            object_states_.begin(), object_states_.end(),
            [&estimate](const ObjectState& state) {
                return state.id == estimate.previous_object_id;
            });
        if (state_it == object_states_.end() || !state_it->has_world_velocity ||
            state_it->age < config_.min_track_age) {
            continue;
        }

        const float translation_speed =
            state_it->world_velocity.template head<3>().norm();
        const float rotation_speed =
            state_it->world_velocity.template tail<3>().norm();
        if (translation_speed < config_.min_object_translation_speed &&
            rotation_speed < config_.min_object_rotation_speed) {
            continue;
        }

        const Matrix6f predicted_velocity_covariance = symmetrized(
            state_it->velocity_covariance +
            dt * velocity_process_covariance);
        const Vector6f predicted_motion_tangent =
            dt * state_it->world_velocity;
        const Matrix6f motion_exp_jacobian =
            leftExpJacobian(predicted_motion_tangent);
        Matrix6f world_motion_covariance = symmetrized(
            motion_exp_jacobian *
            (dt * dt * predicted_velocity_covariance) *
            motion_exp_jacobian.transpose());
        world_motion_covariance.diagonal().array() += 1e-9f;
        const Matrix6f camera_motion_covariance = transformCovariance(
            world_motion_covariance, previous_Tcw_);
        Matrix6f object_prior_information;
        if (!invertPositiveDefinite(
                camera_motion_covariance, object_prior_information)) {
            continue;
        }
        object_prior_information = clampInformation(object_prior_information);

        const Sophus::SE3f object_motion_world =
            Sophus::SE3f::exp(predicted_motion_tangent);
        const Sophus::SE3f object_motion_previous_camera =
            previous_Tcw_ * object_motion_world * previous_Tcw_.inverse();
        const Sophus::SE3f candidate =
            estimate.apparent_transform * object_motion_previous_camera.inverse();
        const float translation = candidate.translation().norm();
        const float rotation = candidate.so3().log().norm();
        if (translation > config_.max_translation || rotation > config_.max_rotation) continue;
        if (has_camera_relative_) {
            const Sophus::SE3f disagreement = candidate * last_camera_relative_.inverse();
            if (disagreement.translation().norm() > config_.max_translation ||
                disagreement.so3().log().norm() > config_.max_rotation) {
                continue;
            }
        }

        const SchurResult marginalized = linearizeAndMarginalize(
            evaluated.correspondences->source,
            evaluated.correspondences->target, evaluated.inliers,
            candidate, object_motion_previous_camera,
            evaluated.residual_variance, object_prior_information);
        if (!marginalized.valid) continue;
        const Sophus::SE3f refined_candidate =
            Sophus::SE3f::exp(marginalized.camera_increment) * candidate;
        if (!refined_candidate.matrix().allFinite() ||
            refined_candidate.translation().norm() > config_.max_translation ||
            refined_candidate.so3().log().norm() > config_.max_rotation) {
            continue;
        }
        if (has_camera_relative_) {
            const Sophus::SE3f disagreement =
                refined_candidate * last_camera_relative_.inverse();
            if (disagreement.translation().norm() > config_.max_translation ||
                disagreement.so3().log().norm() > config_.max_rotation) {
                continue;
            }
        }
        const SchurResult refined_marginalized = linearizeAndMarginalize(
            evaluated.correspondences->source,
            evaluated.correspondences->target, evaluated.inliers,
            refined_candidate, object_motion_previous_camera,
            evaluated.residual_variance, object_prior_information);
        if (!refined_marginalized.valid) continue;
        const Matrix6f information =
            clampInformation(refined_marginalized.camera_information);
        const float score = minimumInformationEigenvalue(information);
        if (score < config_.min_information) continue;
        camera_candidates.push_back(
            CameraCandidate{refined_candidate, information});
    }

    if (!camera_candidates.empty()) {
        const std::vector<std::size_t> consistent_candidates =
            selectConsistentCandidates(
                camera_candidates,
                config_.candidate_mahalanobis_threshold);
        result.objects_used =
            static_cast<int>(consistent_candidates.size());
        if (consistent_candidates.empty()) {
            cacheCurrentFrame(
                gray, depth_meters, current_mask, timestamp);
            return result;
        }
        const Sophus::SE3f reference =
            has_camera_relative_ ? last_camera_relative_
                                 : camera_candidates[
                                       consistent_candidates.front()].pose;
        Matrix6f fused_information = Matrix6f::Zero();
        Vector6f fused_gradient = Vector6f::Zero();
        for (std::size_t index : consistent_candidates) {
            const CameraCandidate& candidate = camera_candidates[index];
            const Vector6f relative_log =
                (candidate.pose * reference.inverse()).log();
            fused_information += candidate.information;
            fused_gradient.noalias() +=
                candidate.information * relative_log;
        }
        Matrix6f inverse_fused_information;
        if (invertPositiveDefinite(
                fused_information, inverse_fused_information)) {
            const Vector6f fused_increment =
                inverse_fused_information * fused_gradient;
            result.relative_pose =
                Sophus::SE3f::exp(fused_increment) * reference;
            result.information = clampInformation(fused_information);
            result.information_score =
                minimumInformationEigenvalue(result.information);
            result.valid = result.information_score >= config_.min_information;
        }
    }

    cacheCurrentFrame(gray, depth_meters, current_mask, timestamp);
    return result;
}

void MotionMarginalizedPosePrior::commitCameraPose(
    const Sophus::SE3f& Tcw, bool pose_valid)
{
    std::vector<const ApparentEstimate*> ordered_estimates;
    ordered_estimates.reserve(pending_estimates_.size());
    for (const ApparentEstimate& estimate : pending_estimates_) {
        ordered_estimates.push_back(&estimate);
    }
    std::sort(
        ordered_estimates.begin(), ordered_estimates.end(),
        [](const ApparentEstimate* lhs, const ApparentEstimate* rhs) {
            if (lhs->inlier_count != rhs->inlier_count) {
                return lhs->inlier_count > rhs->inlier_count;
            }
            return lhs->inlier_ratio > rhs->inlier_ratio;
        });

    std::unordered_map<int, const ApparentEstimate*> estimate_by_label;
    std::set<int> inherited_previous_ids;
    for (const ApparentEstimate* estimate : ordered_estimates) {
        if (estimate_by_label.count(estimate->current_label) != 0 ||
            !inherited_previous_ids.insert(
                estimate->previous_object_id).second) {
            continue;
        }
        estimate_by_label[estimate->current_label] = estimate;
    }

    std::vector<ObjectState> next_states;
    next_states.reserve(current_components_.size());
    for (const Component& component : current_components_) {
        auto estimate_it = estimate_by_label.find(component.label);
        if (estimate_it == estimate_by_label.end()) {
            next_states.push_back(ObjectState{next_object_id_++, component.bbox});
            continue;
        }

        const ApparentEstimate& estimate = *estimate_it->second;
        auto state_it = std::find_if(
            object_states_.begin(), object_states_.end(),
            [&estimate](const ObjectState& state) {
                return state.id == estimate.previous_object_id;
            });
        if (state_it == object_states_.end()) {
            next_states.push_back(ObjectState{next_object_id_++, component.bbox});
            continue;
        }

        ObjectState state = *state_it;
        state.bbox = component.bbox;
        ++state.age;
        if (pose_valid && previous_pose_valid_ && pending_dt_ > 1e-6) {
            const Sophus::SE3f measured_world_motion =
                Tcw.inverse() * estimate.apparent_transform * previous_Tcw_;

            Matrix6f apparent_observation_covariance;
            if (!invertPositiveDefinite(
                    estimate.observation_information,
                    apparent_observation_covariance)) {
                if (state.has_world_velocity) {
                    const Matrix6f process_covariance = diagonalCovariance(
                        config_.velocity_process_translation_sigma,
                        config_.velocity_process_rotation_sigma);
                    state.velocity_covariance = symmetrized(
                        state.velocity_covariance +
                        static_cast<float>(pending_dt_) *
                            process_covariance);
                }
                next_states.push_back(state);
                continue;
            }
            const Matrix6f measurement_covariance = diagonalCovariance(
                config_.measurement_translation_sigma,
                config_.measurement_rotation_sigma);
            const Matrix6f camera_pose_covariance = diagonalCovariance(
                config_.camera_translation_sigma,
                config_.camera_rotation_sigma);
            const Matrix6f world_motion_tangent_covariance = symmetrized(
                transformCovariance(
                    apparent_observation_covariance + measurement_covariance,
                    Tcw.inverse()) +
                transformCovariance(
                    camera_pose_covariance, Tcw.inverse()) +
                transformCovariance(
                    camera_pose_covariance,
                    Tcw.inverse() * estimate.apparent_transform));
            const float dt = static_cast<float>(pending_dt_);
            const Matrix6f motion_log_jacobian =
                leftLogJacobian(measured_world_motion);
            Matrix6f measurement_velocity_covariance = symmetrized(
                motion_log_jacobian * world_motion_tangent_covariance *
                motion_log_jacobian.transpose() / (dt * dt));
            measurement_velocity_covariance.diagonal().array() += 1e-9f;
            const Vector6f measured_world_velocity =
                measured_world_motion.log() / dt;

            if (state.has_world_velocity) {
                const Matrix6f process_covariance = diagonalCovariance(
                    config_.velocity_process_translation_sigma,
                    config_.velocity_process_rotation_sigma);
                Matrix6f predicted_covariance = symmetrized(
                    state.velocity_covariance + dt * process_covariance);
                Matrix6f predicted_information;
                Matrix6f measurement_information;
                if (invertPositiveDefinite(
                        predicted_covariance, predicted_information) &&
                    invertPositiveDefinite(
                        measurement_velocity_covariance,
                        measurement_information)) {
                    Matrix6f posterior_covariance;
                    if (invertPositiveDefinite(
                            predicted_information + measurement_information,
                            posterior_covariance)) {
                        state.world_velocity = posterior_covariance *
                            (predicted_information * state.world_velocity +
                             measurement_information *
                                 measured_world_velocity);
                        state.velocity_covariance =
                            symmetrized(posterior_covariance);
                    }
                }
            } else {
                state.world_velocity = measured_world_velocity;
                state.velocity_covariance =
                    symmetrized(measurement_velocity_covariance);
                state.has_world_velocity = true;
            }
        } else {
            state.has_world_velocity = false;
            state.world_velocity.setZero();
            state.velocity_covariance.setIdentity();
            state.age = 1;
        }
        next_states.push_back(state);
    }
    object_states_.swap(next_states);

    if (pose_valid && previous_pose_valid_) {
        last_camera_relative_ = Tcw * previous_Tcw_.inverse();
        has_camera_relative_ = true;
    } else if (!pose_valid) {
        has_camera_relative_ = false;
    }
    previous_Tcw_ = Tcw;
    previous_pose_valid_ = pose_valid;
    pending_estimates_.clear();
    pending_dt_ = 0.0;
}

void MotionMarginalizedPosePrior::cacheCurrentFrame(
    const cv::Mat& gray, const cv::Mat& depth, const cv::Mat& dynamic_mask,
    double timestamp)
{
    previous_gray_ = gray.clone();
    previous_depth_ = depth.clone();
    previous_dynamic_mask_ = dynamic_mask.clone();
    previous_timestamp_ = timestamp;
}

void MotionMarginalizedPosePrior::reset()
{
    previous_gray_.release();
    previous_depth_.release();
    previous_dynamic_mask_.release();
    object_states_.clear();
    current_components_.clear();
    pending_estimates_.clear();
    previous_pose_valid_ = false;
    has_camera_relative_ = false;
    previous_timestamp_ = 0.0;
    pending_dt_ = 0.0;
    next_object_id_ = 0;
    rng_.seed(config_.seed);
}

}  // namespace Motion3D
