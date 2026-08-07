#include "Motion3D/DirectRgbdPoseValidator.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Motion3D {

DirectRgbdPoseValidator::DirectRgbdPoseValidator()
    : DirectRgbdPoseValidator(Config{}) {}

DirectRgbdPoseValidator::DirectRgbdPoseValidator(const Config& config)
    : config_(config) {
    if (config_.grid_step <= 0 || config_.min_common_support <= 0 ||
        config_.min_depth <= 0.0f || config_.max_depth <= config_.min_depth ||
        config_.depth_sigma_constant <= 0.0f ||
        config_.depth_sigma_linear < 0.0f ||
        config_.photometric_sigma <= 0.0f ||
        config_.depth_weight < 0.0f || config_.photometric_weight < 0.0f ||
        config_.depth_weight + config_.photometric_weight <= 0.0f) {
        throw std::invalid_argument("Invalid direct RGB-D validator config");
    }
}

bool DirectRgbdPoseValidator::prefersDynamic(
    const Result& result, ScoreMode mode) {
    if (!result.valid) {
        return false;
    }
    switch (mode) {
        case ScoreMode::Photometric:
            return result.dynamic_photometric_score <
                result.static_photometric_score;
        case ScoreMode::Depth:
            return result.dynamic_depth_score < result.static_depth_score;
        case ScoreMode::Combined:
            return result.dynamic_combined_score <
                result.static_combined_score;
    }
    return false;
}

cv::Mat DirectRgbdPoseValidator::toGrayFloat(const cv::Mat& image) {
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

float DirectRgbdPoseValidator::bilinear(
    const cv::Mat& image, float x, float y) {
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float dx = x - x0;
    const float dy = y - y0;
    const float top =
        (1.0f - dx) * image.at<float>(y0, x0) +
        dx * image.at<float>(y0, x0 + 1);
    const float bottom =
        (1.0f - dx) * image.at<float>(y0 + 1, x0) +
        dx * image.at<float>(y0 + 1, x0 + 1);
    return (1.0f - dy) * top + dy * bottom;
}

float DirectRgbdPoseValidator::huber(float normalized_residual) {
    const float absolute = std::abs(normalized_residual);
    return absolute <= 1.0f
        ? 0.5f * absolute * absolute
        : absolute - 0.5f;
}

DirectRgbdPoseValidator::Observation DirectRgbdPoseValidator::observe(
    int x,
    int y,
    float previous_depth,
    float previous_intensity,
    const cv::Mat& current_gray,
    const cv::Mat& current_depth_m,
    const cv::Mat& current_static_mask,
    const Sophus::SE3f& current_from_previous,
    const Intrinsics& intrinsics) const {
    Observation observation;
    const Eigen::Vector3f previous_point(
        (static_cast<float>(x) - intrinsics.cx) * previous_depth /
            intrinsics.fx,
        (static_cast<float>(y) - intrinsics.cy) * previous_depth /
            intrinsics.fy,
        previous_depth);
    const Eigen::Vector3f current_point =
        current_from_previous * previous_point;
    if (!current_point.allFinite() ||
        current_point.z() < config_.min_depth ||
        current_point.z() > config_.max_depth) {
        return observation;
    }

    const float projected_x =
        intrinsics.fx * current_point.x() / current_point.z() + intrinsics.cx;
    const float projected_y =
        intrinsics.fy * current_point.y() / current_point.z() + intrinsics.cy;
    if (!std::isfinite(projected_x) || !std::isfinite(projected_y) ||
        projected_x < 0.0f || projected_y < 0.0f ||
        projected_x >= current_gray.cols - 1.0f ||
        projected_y >= current_gray.rows - 1.0f) {
        return observation;
    }

    const int nearest_x = static_cast<int>(std::lround(projected_x));
    const int nearest_y = static_cast<int>(std::lround(projected_y));
    if (nearest_x < 0 || nearest_x >= current_gray.cols ||
        nearest_y < 0 || nearest_y >= current_gray.rows ||
        current_static_mask.at<unsigned char>(nearest_y, nearest_x) == 0) {
        return observation;
    }
    const float observed_depth =
        current_depth_m.at<float>(nearest_y, nearest_x);
    if (!std::isfinite(observed_depth) ||
        observed_depth < config_.min_depth ||
        observed_depth > config_.max_depth) {
        return observation;
    }
    const float current_intensity =
        bilinear(current_gray, projected_x, projected_y);
    if (!std::isfinite(current_intensity)) {
        return observation;
    }

    const float depth_sigma = config_.depth_sigma_constant +
        config_.depth_sigma_linear * current_point.z();
    observation.depth_cost = huber(
        (observed_depth - current_point.z()) / depth_sigma);
    observation.photometric_cost = huber(
        (current_intensity - previous_intensity) /
        config_.photometric_sigma);
    observation.valid = std::isfinite(observation.depth_cost) &&
        std::isfinite(observation.photometric_cost);
    return observation;
}

DirectRgbdPoseValidator::Result DirectRgbdPoseValidator::score(
    const cv::Mat& previous_bgr_or_gray,
    const cv::Mat& current_bgr_or_gray,
    const cv::Mat& previous_depth_m,
    const cv::Mat& current_depth_m,
    const cv::Mat& previous_static_mask,
    const cv::Mat& current_static_mask,
    const Sophus::SE3f& previous_tcw,
    const Sophus::SE3f& static_tcw,
    const Sophus::SE3f& dynamic_tcw,
    const Intrinsics& intrinsics) const {
    Result result;
    if (intrinsics.fx <= 0.0f || intrinsics.fy <= 0.0f ||
        !previous_tcw.matrix().allFinite() ||
        !static_tcw.matrix().allFinite() ||
        !dynamic_tcw.matrix().allFinite() ||
        previous_depth_m.type() != CV_32FC1 ||
        current_depth_m.type() != CV_32FC1 ||
        previous_static_mask.type() != CV_8UC1 ||
        current_static_mask.type() != CV_8UC1) {
        return result;
    }

    const cv::Mat previous_gray = toGrayFloat(previous_bgr_or_gray);
    const cv::Mat current_gray = toGrayFloat(current_bgr_or_gray);
    if (previous_gray.empty() || current_gray.empty()) {
        return result;
    }
    const cv::Size size = previous_gray.size();
    if (current_gray.size() != size || previous_depth_m.size() != size ||
        current_depth_m.size() != size || previous_static_mask.size() != size ||
        current_static_mask.size() != size) {
        return result;
    }

    const Sophus::SE3f static_from_previous =
        static_tcw * previous_tcw.inverse();
    const Sophus::SE3f dynamic_from_previous =
        dynamic_tcw * previous_tcw.inverse();
    double static_depth = 0.0;
    double dynamic_depth = 0.0;
    double static_photometric = 0.0;
    double dynamic_photometric = 0.0;
    for (int y = config_.grid_step / 2;
         y < size.height;
         y += config_.grid_step) {
        for (int x = config_.grid_step / 2;
             x < size.width;
             x += config_.grid_step) {
            if (previous_static_mask.at<unsigned char>(y, x) == 0) {
                continue;
            }
            const float depth = previous_depth_m.at<float>(y, x);
            if (!std::isfinite(depth) || depth < config_.min_depth ||
                depth > config_.max_depth) {
                continue;
            }
            const float intensity = previous_gray.at<float>(y, x);
            if (!std::isfinite(intensity)) {
                continue;
            }
            const Observation static_observation = observe(
                x, y, depth, intensity, current_gray, current_depth_m,
                current_static_mask, static_from_previous, intrinsics);
            const Observation dynamic_observation = observe(
                x, y, depth, intensity, current_gray, current_depth_m,
                current_static_mask, dynamic_from_previous, intrinsics);
            if (!static_observation.valid || !dynamic_observation.valid) {
                continue;
            }
            static_depth += static_observation.depth_cost;
            dynamic_depth += dynamic_observation.depth_cost;
            static_photometric += static_observation.photometric_cost;
            dynamic_photometric += dynamic_observation.photometric_cost;
            ++result.common_support;
        }
    }

    if (result.common_support == 0) {
        return result;
    }
    const float inverse_support = 1.0f / result.common_support;
    result.static_depth_score = static_cast<float>(static_depth) * inverse_support;
    result.dynamic_depth_score =
        static_cast<float>(dynamic_depth) * inverse_support;
    result.static_photometric_score =
        static_cast<float>(static_photometric) * inverse_support;
    result.dynamic_photometric_score =
        static_cast<float>(dynamic_photometric) * inverse_support;
    const float total_weight =
        config_.depth_weight + config_.photometric_weight;
    result.static_combined_score = (
        config_.depth_weight * result.static_depth_score +
        config_.photometric_weight * result.static_photometric_score) /
        total_weight;
    result.dynamic_combined_score = (
        config_.depth_weight * result.dynamic_depth_score +
        config_.photometric_weight * result.dynamic_photometric_score) /
        total_weight;
    result.valid = result.common_support >= config_.min_common_support;
    return result;
}

}  // namespace Motion3D
