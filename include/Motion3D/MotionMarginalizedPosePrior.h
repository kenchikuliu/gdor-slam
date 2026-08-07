#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <cstddef>
#include <random>
#include <unordered_map>
#include <vector>

namespace Motion3D {

class MotionMarginalizedPosePrior {
public:
    using Matrix6f = Eigen::Matrix<float, 6, 6>;
    using Vector6f = Eigen::Matrix<float, 6, 1>;

    struct CameraIntrinsics {
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
    };

    struct Config {
        int max_features = 400;
        int min_features_per_object = 20;
        int min_component_area = 400;
        int min_track_age = 2;
        int ransac_iterations = 128;
        float ransac_threshold = 0.05f;
        float min_inlier_ratio = 0.5f;
        float min_information = 5.0f;
        float max_translation = 0.5f;
        float max_rotation = 0.5236f;
        float lk_forward_backward_threshold = 1.5f;
        bool use_depth_foreground_filter = true;
        float foreground_depth_separation = 0.15f;
        float foreground_min_fraction = 0.08f;
        float min_object_translation_speed = 0.03f;
        float min_object_rotation_speed = 0.05f;
        float measurement_translation_sigma = 0.01f;
        float measurement_rotation_sigma = 0.02f;
        float camera_translation_sigma = 0.02f;
        float camera_rotation_sigma = 0.02f;
        float velocity_process_translation_sigma = 0.20f;
        float velocity_process_rotation_sigma = 0.35f;
        float max_information_eigenvalue = 1e4f;
        float candidate_mahalanobis_threshold = 16.812f;
        unsigned int seed = 0;
    };

    struct SchurResult {
        bool valid = false;
        Matrix6f Hcc = Matrix6f::Zero();
        Matrix6f Hco = Matrix6f::Zero();
        Matrix6f Hoo = Matrix6f::Zero();
        Vector6f bc = Vector6f::Zero();
        Vector6f bo = Vector6f::Zero();
        Matrix6f camera_information = Matrix6f::Zero();
        Vector6f camera_gradient = Vector6f::Zero();
        Vector6f camera_increment = Vector6f::Zero();
    };

    struct CameraCandidate {
        Sophus::SE3f pose;
        Matrix6f information = Matrix6f::Zero();
    };

    struct Result {
        bool valid = false;
        Sophus::SE3f relative_pose;
        Matrix6f information = Matrix6f::Zero();
        float information_score = 0.0f;
        int objects_observed = 0;
        int objects_used = 0;
        int feature_tracks = 0;
        int inlier_tracks = 0;
        float foreground_fraction = 0.0f;
    };

    MotionMarginalizedPosePrior(const Config& config,
                                const CameraIntrinsics& intrinsics);

    Result prepare(const cv::Mat& rgb,
                   const cv::Mat& depth_meters,
                   const cv::Mat& semantic_dynamic_mask,
                   double timestamp);

    void commitCameraPose(const Sophus::SE3f& Tcw, bool pose_valid);
    void reset();

    static bool estimateRigidTransform(
        const std::vector<Eigen::Vector3f>& source,
        const std::vector<Eigen::Vector3f>& target,
        float inlier_threshold,
        int ransac_iterations,
        std::mt19937& rng,
        Sophus::SE3f& transform,
        std::vector<unsigned char>& inliers,
        float& rmse);

    static SchurResult linearizeAndMarginalize(
        const std::vector<Eigen::Vector3f>& source,
        const std::vector<Eigen::Vector3f>& target,
        const std::vector<unsigned char>& inliers,
        const Sophus::SE3f& camera_motion,
        const Sophus::SE3f& object_motion,
        float residual_variance,
        const Matrix6f& object_prior_information);

    static float minimumInformationEigenvalue(const Matrix6f& information);
    static Matrix6f transformCovariance(
        const Matrix6f& covariance,
        const Sophus::SE3f& target_from_source);
    static Matrix6f leftExpJacobian(const Vector6f& tangent);
    static Matrix6f leftLogJacobian(const Sophus::SE3f& transform);
    static float candidateInnovationMahalanobis(
        const CameraCandidate& lhs,
        const CameraCandidate& rhs);
    static std::vector<std::size_t> selectConsistentCandidates(
        const std::vector<CameraCandidate>& candidates,
        float mahalanobis_threshold);

    std::vector<int> activeObjectIds() const;

private:
    struct Component {
        int label = 0;
        cv::Rect bbox;
        int area = 0;
    };

    struct ObjectState {
        int id = -1;
        cv::Rect bbox;
        int age = 1;
        bool has_world_velocity = false;
        Vector6f world_velocity = Vector6f::Zero();
        Matrix6f velocity_covariance = Matrix6f::Identity();
    };

    struct ApparentEstimate {
        int previous_object_id = -1;
        int current_label = 0;
        cv::Rect current_bbox;
        Sophus::SE3f apparent_transform;
        Matrix6f observation_information = Matrix6f::Zero();
        int track_count = 0;
        int inlier_count = 0;
        float inlier_ratio = 0.0f;
        float rmse = 0.0f;
    };

    Config config_;
    CameraIntrinsics intrinsics_;
    std::mt19937 rng_;

    cv::Mat previous_gray_;
    cv::Mat previous_depth_;
    cv::Mat previous_dynamic_mask_;
    std::vector<ObjectState> object_states_;
    std::vector<Component> current_components_;
    std::vector<ApparentEstimate> pending_estimates_;

    Sophus::SE3f previous_Tcw_;
    Sophus::SE3f last_camera_relative_;
    bool previous_pose_valid_ = false;
    bool has_camera_relative_ = false;
    double previous_timestamp_ = 0.0;
    double pending_dt_ = 0.0;
    int next_object_id_ = 0;

    cv::Mat refineForegroundMask(const cv::Mat& semantic_mask,
                                 const cv::Mat& depth) const;
    std::vector<Component> extractComponents(const cv::Mat& dynamic_mask,
                                             cv::Mat& labels) const;
    std::vector<int> matchStatesToComponents(
        const std::vector<Component>& components) const;
    Eigen::Vector3f backProject(const cv::Point2f& pixel, float depth) const;
    Matrix6f diagonalCovariance(float translation_sigma,
                                float rotation_sigma) const;
    Matrix6f clampInformation(const Matrix6f& information) const;
    void cacheCurrentFrame(const cv::Mat& gray,
                           const cv::Mat& depth,
                           const cv::Mat& dynamic_mask,
                           double timestamp);
};

}  // namespace Motion3D
