#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <vector>

namespace Motion3D {

/**
 * @brief Selects conservative static background support around dynamic masks.
 *
 * The router is deliberately an intervention-free primitive.  It never
 * changes either the hard tracking mask or the Gaussian mapping mask.  It
 * only returns a deterministic set of pixels that may be used by a later
 * pose-proposal stage after independent validation.
 */
class BackgroundSupportRouter {
public:
    struct Config {
        int ring_radius = 3;
        int sample_stride = 4;
        int max_support_points = 256;
        int min_support_points = 40;
        float min_depth = 0.1f;
        float max_depth = 8.0f;
        float max_depth_residual = 0.05f;
        float depth_residual_scale = 0.02f;
        float min_gradient = 0.02f;
        int flow_guard_radius = 1;
        bool require_rendered_depth = true;
    };

    struct SupportPoint {
        cv::Point pixel;
        float depth = 0.0f;
        float depth_residual = 0.0f;
        float gradient = 0.0f;
        float score = 0.0f;
    };

    struct Result {
        bool valid = false;
        int dynamic_pixels = 0;
        int ring_pixels = 0;
        int candidate_points = 0;
        int selected_points = 0;
        float support_ratio = 0.0f;
        float median_depth_residual = -1.0f;
        cv::Mat ring_mask;
        cv::Mat support_mask;
        std::vector<SupportPoint> points;
    };

    BackgroundSupportRouter();
    explicit BackgroundSupportRouter(const Config& config);

    const Config& config() const { return config_; }

    /**
     * @param current_gray_or_bgr Current image, CV_8U/CV_32F gray or BGR.
     * @param depth_meters Current aligned depth, CV_32FC1.
     * @param dynamic_mask Binary dynamic mask (non-zero means dynamic).
     * @param static_mask Binary tracking mask (non-zero means static).
     * @param rendered_depth Optional background depth rendered at the parent
     *        pose.  It is required by default.
     * @param flow_guard Optional residual-flow mask.  Non-zero pixels veto
     *        support within config.flow_guard_radius.
     */
    Result select(
        const cv::Mat& current_gray_or_bgr,
        const cv::Mat& depth_meters,
        const cv::Mat& dynamic_mask,
        const cv::Mat& static_mask,
        const cv::Mat& rendered_depth = cv::Mat(),
        const cv::Mat& flow_guard = cv::Mat()) const;

private:
    static cv::Mat toGrayFloat(const cv::Mat& image);
    static cv::Mat toBinaryMask(const cv::Mat& mask, const cv::Size& size);
    static float median(std::vector<float> values);

    Config config_;
};

/**
 * @brief Shadow-only RGB-D pose proposal driven by routed background support.
 *
 * This class performs a small robust point-to-point refinement around the
 * parent relative pose.  It does not apply the proposal to a tracker or map;
 * callers must independently verify the returned candidate and explicitly
 * choose whether to use it.
 */
class BackgroundSupportPoseRefiner {
public:
    using Matrix6f = Eigen::Matrix<float, 6, 6>;

    struct Intrinsics {
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
    };

    struct Config {
        BackgroundSupportRouter::Config router;
        int max_iterations = 4;
        float huber_delta = 0.03f;
        float min_inlier_ratio = 0.5f;
        float min_rmse_improvement = 0.01f;
        float max_translation_correction = 0.05f;
        float max_rotation_correction = 0.15f;
        bool translation_only = true;
    };

    struct Result {
        bool valid = false;
        bool improved = false;
        Sophus::SE3f refined_relative_pose;
        Matrix6f information = Matrix6f::Zero();
        int routed_support = 0;
        int geometric_support = 0;
        int inliers = 0;
        float initial_rmse = -1.0f;
        float final_rmse = -1.0f;
        float translation_correction = 0.0f;
        float rotation_correction = 0.0f;
    };

    BackgroundSupportPoseRefiner(
        const Config& config, const Intrinsics& intrinsics);

    const Config& config() const { return config_; }

    Result refine(
        const cv::Mat& previous_gray_or_bgr,
        const cv::Mat& current_gray_or_bgr,
        const cv::Mat& previous_depth_meters,
        const cv::Mat& current_depth_meters,
        const cv::Mat& previous_static_mask,
        const cv::Mat& current_static_mask,
        const cv::Mat& current_dynamic_mask,
        const cv::Mat& current_rendered_depth,
        const cv::Mat& current_flow_guard,
        const Sophus::SE3f& parent_relative_pose) const;

private:
    struct Correspondence {
        Eigen::Vector3f previous;
        Eigen::Vector3f current;
    };

    static Eigen::Matrix3f skew(const Eigen::Vector3f& value);
    static float huberWeight(float residual_norm, float delta);
    static bool project(
        const Eigen::Vector3f& point,
        const Intrinsics& intrinsics,
        const cv::Size& image_size,
        cv::Point2f& pixel);
    static float rmse(
        const std::vector<Correspondence>& correspondences,
        const Sophus::SE3f& relative_pose,
        float inlier_threshold,
        int* inliers);

    Config config_;
    Intrinsics intrinsics_;
    BackgroundSupportRouter router_;
};

/**
 * @brief Translation-only pose rescue against a persistent Gaussian depth map.
 *
 * Model-depth pixels are deterministically split into disjoint proposal and
 * validation tiles.  The proposal half drives projective point-to-plane ICP;
 * the validation half only decides whether the candidate is admissible.  The
 * class is side-effect free and never mutates a tracker, MapPoint, or Gaussian.
 */
class GaussianBackgroundPoseRefiner {
public:
    struct Intrinsics {
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
    };

    struct Config {
        int sample_stride = 4;
        int split_tile_size = 16;
        int dynamic_dilation_radius = 8;
        int max_iterations = 4;
        int min_proposal_support = 80;
        int min_validation_support = 80;
        int minimum_observable_rank = 2;
        float min_depth = 0.1f;
        float max_depth = 8.0f;
        float max_correspondence_distance = 0.10f;
        float huber_delta = 0.03f;
        float min_proposal_relative_improvement = 0.01f;
        float min_validation_relative_improvement = 0.01f;
        float eigenvalue_relative_threshold = 1e-3f;
        float max_translation_correction = 0.02f;
    };

    struct Result {
        bool candidate_available = false;
        bool proposal_improved = false;
        bool validation_valid = false;
        bool validation_prefers_candidate = false;
        bool gate_pass = false;
        Sophus::SE3f refined_tcw;
        int dynamic_pixels = 0;
        int safe_static_pixels = 0;
        int proposal_points = 0;
        int validation_points = 0;
        int proposal_validation_overlap = 0;
        int proposal_support = 0;
        int validation_common_support = 0;
        int observable_rank = 0;
        float proposal_parent_loss = -1.0f;
        float proposal_candidate_loss = -1.0f;
        float validation_parent_loss = -1.0f;
        float validation_candidate_loss = -1.0f;
        float translation_correction = 0.0f;
        float rotation_correction = 0.0f;
    };

    GaussianBackgroundPoseRefiner(
        const Config& config, const Intrinsics& intrinsics);

    const Config& config() const { return config_; }

    Result refine(
        const cv::Mat& current_depth_meters,
        const cv::Mat& static_mask,
        const cv::Mat& dynamic_mask,
        const cv::Mat& gaussian_depth_at_parent,
        const Sophus::SE3f& parent_tcw) const;

private:
    struct ModelPoint {
        cv::Point pixel;
        Eigen::Vector3f point;
        Eigen::Vector3f normal;
    };

    struct Observation {
        bool valid = false;
        float residual = 0.0f;
    };

    struct Score {
        bool valid = false;
        int common_support = 0;
        float parent_loss = -1.0f;
        float candidate_loss = -1.0f;
    };

    bool finiteDepth(float depth) const;
    Eigen::Vector3f unproject(float x, float y, float depth) const;
    bool modelNormal(
        const cv::Mat& depth,
        int x,
        int y,
        Eigen::Vector3f& normal) const;
    Observation observe(
        const ModelPoint& model,
        const Eigen::Vector3f& translation,
        const cv::Mat& current_depth,
        const cv::Mat& safe_static_mask) const;
    std::vector<ModelPoint> collectModelPoints(
        const cv::Mat& gaussian_depth,
        const cv::Mat& safe_static_mask,
        int parity) const;
    Score scoreCommon(
        const std::vector<ModelPoint>& points,
        const Eigen::Vector3f& candidate_translation,
        const cv::Mat& current_depth,
        const cv::Mat& safe_static_mask,
        int minimum_support) const;
    static float huberWeight(float residual, float delta);
    static float huberLoss(float residual, float delta);

    Config config_;
    Intrinsics intrinsics_;
};

}  // namespace Motion3D
