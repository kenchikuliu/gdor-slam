#pragma once

#include "Motion3D/DirectRgbdPoseValidator.h"
#include "Motion3D/ReplayStateContract.h"

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <sophus/se3.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace Motion3D {

static constexpr const char* kFrozenReplayPacketVersion =
    "frozen-same-state-replay-v4";
static constexpr std::uint64_t kNoReplayMapPoint =
    static_cast<std::uint64_t>(-1);
static constexpr int kReplayFreezeSettlingFrames = 1;
static constexpr float kReplayScoreAbsoluteTolerance = 1e-5f;
static constexpr float kReplayScoreRelativeTolerance = 1e-5f;
static constexpr float kReplayDirectScoreAbsoluteTolerance = 1e-4f;
static constexpr float kReplayPoseTranslationTolerance = 5e-5f;
static constexpr float kReplayPoseRotationTolerance = 1e-5f;

enum class ReplayGatePolicy {
    LegacyConjunction = 0,
    DirectCombined = 1,
};

const char* replayGatePolicyName(ReplayGatePolicy policy);
bool parseReplayGatePolicy(
    const std::string& name, ReplayGatePolicy& policy);

struct ReplayGateConfig {
    ReplayGatePolicy policy = ReplayGatePolicy::LegacyConjunction;
    int min_inlier_gain = 5;
    float min_inlier_gain_ratio = 0.05f;
    int max_static_inliers = 2147483647;
    float min_translation_innovation = 0.0f;
    float max_translation_innovation = 1e6f;
    float max_rotation_innovation = 3.1415927f;
    bool use_direct_validation = false;
    int direct_score_mode = 0;
    bool require_common_support_improvement = false;
    bool bypass_reliability_gate = false;
};

struct FrozenReplayMapPoint {
    std::uint64_t id = 0;
    Eigen::Vector3f world_position = Eigen::Vector3f::Zero();
    int observations = 0;
    cv::Mat descriptor;
};

struct FrozenReplayBranchResult {
    int matches = -1;
    int inliers = -1;
    Sophus::SE3f optimized_tcw;
};

enum class TranslationLeverageMode {
    CapOnly = 0,
    NormalizeToTarget = 1,
};

const char* translationLeverageModeName(TranslationLeverageMode mode);
bool parseTranslationLeverageMode(
    const std::string& name, TranslationLeverageMode& mode);

struct TranslationLeverageDiagnostics {
    bool valid = false;
    float static_translation_trace = 0.0f;
    float dynamic_translation_trace_raw = 0.0f;
    float dynamic_translation_trace_bounded = 0.0f;
    float max_generalized_leverage_raw = 0.0f;
    float max_generalized_leverage_bounded = 0.0f;
    float normalization_scale = 1.0f;
};

struct FrozenReplayPacket {
    bool valid = false;
    std::string packet_version = kFrozenReplayPacketVersion;
    std::string hash_algorithm = kReplayStateHashAlgorithm;
    std::string branch_type = "motion_model_projection";
    std::string previous_image_name;
    std::string current_image_name;

    ReplayStateIdentity state_identity;
    ReplayFreezeState freeze;
    double previous_timestamp = 0.0;
    double current_timestamp = 0.0;
    Sophus::SE3f previous_tcw;
    Sophus::SE3f velocity;
    Sophus::SE3f static_initial_tcw;
    Sophus::SE3f dynamic_initial_tcw;
    Eigen::Matrix<float, 6, 6> dynamic_pose_information =
        Eigen::Matrix<float, 6, 6>::Identity();
    float dynamic_information_scale = 1.0f;
    bool dynamic_initialization_only = false;

    int search_threshold = 15;
    int minimum_matches = 20;
    int feature_count = 0;
    int previous_feature_count = 0;
    int scale_levels = 0;
    float scale_factor = 1.0f;
    float baseline = 0.0f;
    float baseline_fx = 0.0f;
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float min_x = 0.0f;
    float max_x = 0.0f;
    float min_y = 0.0f;
    float max_y = 0.0f;
    float grid_width_inverse = 0.0f;
    float grid_height_inverse = 0.0f;

    std::vector<float> scale_factors;
    std::vector<float> inverse_level_sigma2;
    std::vector<cv::KeyPoint> current_keypoints;
    std::vector<cv::KeyPoint> previous_keypoints;
    std::vector<float> current_right_coordinates;
    cv::Mat current_descriptors;
    std::vector<std::uint64_t> previous_map_point_ids;
    std::vector<unsigned char> previous_outliers;
    std::vector<FrozenReplayMapPoint> map_points;
    bool local_map_snapshot_valid = false;
    std::vector<std::uint64_t> local_map_point_ids;
    std::vector<unsigned char> local_map_outliers;
    std::uint64_t local_map_support_hash = 0;

    cv::Mat previous_gray;
    cv::Mat current_gray;
    cv::Mat previous_depth;
    cv::Mat current_depth;
    cv::Mat previous_static_mask;
    cv::Mat current_static_mask;
    std::uint64_t previous_gray_hash = 0;
    std::uint64_t current_gray_hash = 0;
    std::uint64_t previous_depth_hash = 0;
    std::uint64_t current_depth_hash = 0;
    std::uint64_t previous_static_mask_hash = 0;
    std::uint64_t current_static_mask_hash = 0;

    ReplayGateConfig gate;
    DirectRgbdPoseValidator::Config direct_config;
    FrozenReplayBranchResult expected_static;
    FrozenReplayBranchResult expected_dynamic;
    int expected_common_support = 0;
    float expected_static_common_score = -1.0f;
    float expected_dynamic_common_score = -1.0f;
    DirectRgbdPoseValidator::Result expected_direct;
    bool expected_consensus_pass = false;
    bool expected_direct_pass = false;
    bool expected_common_support_pass = false;
    bool expected_would_use = false;

    float shadow_translation_blend = 1.0f;
    float posterior_min_score_improvement = 0.0f;
    bool posterior_evaluated = false;
    Sophus::SE3f posterior_static_initial_tcw;
    Sophus::SE3f posterior_dynamic_prior_tcw;
    Eigen::Matrix<float, 6, 6> posterior_dynamic_information =
        Eigen::Matrix<float, 6, 6>::Zero();
    FrozenReplayBranchResult expected_posterior_static;
    FrozenReplayBranchResult expected_posterior_dynamic;
    int expected_posterior_common_support = 0;
    float expected_posterior_static_score = -1.0f;
    float expected_posterior_dynamic_score = -1.0f;
    bool expected_posterior_valid = false;
    bool expected_posterior_pass = false;
};

struct FrozenReplayResult {
    bool valid = false;
    std::string error;
    FrozenReplayBranchResult static_branch;
    FrozenReplayBranchResult dynamic_branch;
    int common_support = 0;
    float static_common_score = -1.0f;
    float dynamic_common_score = -1.0f;
    DirectRgbdPoseValidator::Result direct;
    bool consensus_pass = false;
    bool direct_pass = false;
    bool common_support_pass = false;
    bool would_use = false;
    std::uint64_t static_support_hash = 0;
    std::uint64_t dynamic_support_hash = 0;
    std::uint64_t common_support_hash = 0;
};

struct FrozenReplayFrameState {
    bool valid = false;
    Sophus::SE3f tcw;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<std::uint64_t> map_point_ids;
    std::vector<unsigned char> outliers;
    std::vector<FrozenReplayMapPoint> map_points;
};

struct FrozenReplayBranchStateResult {
    bool valid = false;
    std::string error;
    FrozenReplayBranchResult branch;
    FrozenReplayFrameState frame;
};

struct FrozenReplayPosteriorResult {
    bool valid = false;
    std::string error;
    FrozenReplayBranchResult static_branch;
    FrozenReplayBranchResult dynamic_branch;
    int common_support = 0;
    float static_score = -1.0f;
    float dynamic_score = -1.0f;
    bool gate_valid = false;
    bool pass = false;
    TranslationLeverageDiagnostics leverage;
    FrozenReplayFrameState static_frame;
    FrozenReplayFrameState dynamic_frame;
};

bool evaluateConsensusGate(
    const ReplayGateConfig& config,
    int static_inliers,
    int dynamic_inliers,
    const Sophus::SE3f& static_tcw,
    const Sophus::SE3f& dynamic_tcw);
bool evaluateReplayGatePolicy(
    const ReplayGateConfig& config,
    const Sophus::SE3f& static_tcw,
    const Sophus::SE3f& dynamic_tcw,
    const DirectRgbdPoseValidator::Result& direct,
    bool legacy_consensus_pass,
    bool legacy_direct_pass,
    bool legacy_common_support_pass);
void projectPosePriorToTranslationSubspace(
    const Sophus::SE3f& reference_tcw,
    Sophus::SE3f& prior_tcw,
    Eigen::Matrix<float, 6, 6>& information);
Sophus::SE3f projectCameraInterventionToTranslation(
    const Sophus::SE3f& static_tcw,
    const Sophus::SE3f& dynamic_tcw);
Sophus::SE3f reuseShadowCameraTranslation(
    const Sophus::SE3f& live_tcw,
    const Sophus::SE3f& shadow_tcw,
    float translation_blend = 1.0f);
Eigen::Matrix<float, 6, 6> isotropicTranslationInformation(
    const Eigen::Matrix<float, 6, 6>& information);
Eigen::Matrix<float, 6, 6> leverageBoundedTranslationInformation(
    const Eigen::Matrix<float, 6, 6>& static_information,
    const Eigen::Matrix<float, 6, 6>& dynamic_information,
    float max_generalized_leverage,
    TranslationLeverageDiagnostics* diagnostics = nullptr);
Eigen::Matrix<float, 6, 6> leverageCalibratedTranslationInformation(
    const Eigen::Matrix<float, 6, 6>& static_information,
    const Eigen::Matrix<float, 6, 6>& dynamic_information,
    float leverage_value,
    TranslationLeverageMode mode,
    TranslationLeverageDiagnostics* diagnostics = nullptr);
Sophus::SE3f updateCameraVelocity(
    const Sophus::SE3f& incoming_velocity,
    const Sophus::SE3f& previous_tcw,
    const Sophus::SE3f& current_tcw,
    bool velocity_neutral);
bool posteriorLocalMapGate(
    int static_inliers,
    int dynamic_inliers,
    float static_score,
    float dynamic_score,
    float minimum_score_improvement);
inline float cameraCenterTranslationInnovation(
    const Sophus::SE3f& first_tcw,
    const Sophus::SE3f& second_tcw) {
    return (
        first_tcw.inverse().translation() -
        second_tcw.inverse().translation()).norm();
}
bool replayScoreEquivalent(float expected, float actual);
bool replayDirectScoreEquivalent(float expected, float actual);
bool replayPacketPastFreezeSettlingFrame(
    int frame_index,
    int freeze_map_after_frame);

bool writeFrozenReplayPacket(
    const FrozenReplayPacket& packet,
    const std::string& path,
    std::string* error = nullptr);
bool readFrozenReplayPacket(
    const std::string& path,
    FrozenReplayPacket& packet,
    std::string* error = nullptr);
FrozenReplayResult runFrozenReplay(const FrozenReplayPacket& packet);
FrozenReplayBranchStateResult runFrozenReplayBranch(
    const FrozenReplayPacket& packet,
    const FrozenReplayFrameState& previous,
    const Sophus::SE3f& initial_tcw,
    bool apply_dynamic_prior = false);
FrozenReplayPosteriorResult runFrozenReplayPosterior(
    const FrozenReplayPacket& packet,
    const Sophus::SE3f& static_initial_tcw,
    const Sophus::SE3f& dynamic_prior_tcw,
    const Eigen::Matrix<float, 6, 6>& dynamic_information,
    float dynamic_information_scale,
    float minimum_score_improvement,
    float max_static_information_leverage = 0.0f,
    TranslationLeverageMode leverage_mode =
        TranslationLeverageMode::CapOnly,
    bool compute_dynamic = true);
FrozenReplayPosteriorResult runFrozenReplayPosterior(
    const FrozenReplayPacket& packet);

}  // namespace Motion3D
