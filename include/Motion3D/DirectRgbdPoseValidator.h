#pragma once

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

namespace Motion3D {

class DirectRgbdPoseValidator {
public:
    enum class ScoreMode {
        Photometric = 0,
        Depth = 1,
        Combined = 2,
    };

    struct Intrinsics {
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
    };

    struct Config {
        int grid_step = 8;
        int min_common_support = 50;
        float min_depth = 0.1f;
        float max_depth = 8.0f;
        float depth_sigma_constant = 0.01f;
        float depth_sigma_linear = 0.01f;
        float photometric_sigma = 0.05f;
        float depth_weight = 1.0f;
        float photometric_weight = 1.0f;
    };

    struct Result {
        bool valid = false;
        int common_support = 0;
        float static_depth_score = -1.0f;
        float dynamic_depth_score = -1.0f;
        float static_photometric_score = -1.0f;
        float dynamic_photometric_score = -1.0f;
        float static_combined_score = -1.0f;
        float dynamic_combined_score = -1.0f;
    };

    DirectRgbdPoseValidator();
    explicit DirectRgbdPoseValidator(const Config& config);
    const Config& config() const { return config_; }

    static bool prefersDynamic(const Result& result, ScoreMode mode);

    Result score(
        const cv::Mat& previous_bgr_or_gray,
        const cv::Mat& current_bgr_or_gray,
        const cv::Mat& previous_depth_m,
        const cv::Mat& current_depth_m,
        const cv::Mat& previous_static_mask,
        const cv::Mat& current_static_mask,
        const Sophus::SE3f& previous_tcw,
        const Sophus::SE3f& static_tcw,
        const Sophus::SE3f& dynamic_tcw,
        const Intrinsics& intrinsics) const;

private:
    struct Observation {
        bool valid = false;
        float depth_cost = 0.0f;
        float photometric_cost = 0.0f;
    };

    static cv::Mat toGrayFloat(const cv::Mat& image);
    static float bilinear(const cv::Mat& image, float x, float y);
    static float huber(float normalized_residual);

    Observation observe(
        int x,
        int y,
        float previous_depth,
        float previous_intensity,
        const cv::Mat& current_gray,
        const cv::Mat& current_depth_m,
        const cv::Mat& current_static_mask,
        const Sophus::SE3f& current_from_previous,
        const Intrinsics& intrinsics) const;

    Config config_;
};

}  // namespace Motion3D
