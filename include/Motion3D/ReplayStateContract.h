#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace Motion3D {

static constexpr const char* kReplayStateContractVersion =
    "frozen-replay-state-v2";
static constexpr const char* kReplayStateHashAlgorithm = "fnv1a64-le-v1";
static constexpr const char* kReplayFreezeProtocolVersion =
    "deterministic-map-freeze-v2";

struct ReplayFreezeState {
    std::string protocol = kReplayFreezeProtocolVersion;
    bool requested = false;
    bool local_mapping_idle_ack = false;
    bool local_mapping_stopped_ack = false;
    bool loop_closing_idle_ack = false;
    bool loop_closing_stopped_ack = false;
    bool gba_stopped_ack = false;
    std::uint64_t epoch = 0;

    bool complete() const {
        return requested &&
            local_mapping_idle_ack &&
            local_mapping_stopped_ack &&
            loop_closing_idle_ack &&
            loop_closing_stopped_ack &&
            gba_stopped_ack;
    }
};

struct ReplayMapPointState {
    std::uint64_t map_point_id = 0;
    Eigen::Vector3f world_position = Eigen::Vector3f::Zero();
};

struct ReplayMapPointTrackingState {
    std::uint64_t map_point_id = 0;
    Eigen::Vector3f world_position = Eigen::Vector3f::Zero();
    Eigen::Vector3f normal = Eigen::Vector3f::Zero();
    float min_distance = 0.0f;
    float max_distance = 0.0f;
    std::uint64_t descriptor_hash = 0;
    std::int32_t observations = 0;
};

struct ReplayMapPointTrackingDigest {
    std::uint64_t state_hash = 0;
    std::uint64_t id_hash = 0;
    std::uint64_t position_hash = 0;
    std::uint64_t normal_hash = 0;
    std::uint64_t distance_hash = 0;
    std::uint64_t descriptor_hash = 0;
    std::uint64_t observation_hash = 0;
};

struct ReplayKeyFrameState {
    std::uint64_t keyframe_id = 0;
    Sophus::SE3f tcw;
};

struct ReplaySupportObservation {
    std::uint64_t feature_index = 0;
    std::uint64_t map_point_id = 0;
    Eigen::Vector3f world_position = Eigen::Vector3f::Zero();
    float keypoint_x = 0.0f;
    float keypoint_y = 0.0f;
    float right_coordinate = -1.0f;
    std::int32_t octave = -1;
};

struct ReplayStateIdentity {
    bool valid = false;
    std::string contract_version = kReplayStateContractVersion;
    std::string hash_algorithm = kReplayStateHashAlgorithm;
    std::uint64_t previous_frame_id = 0;
    std::uint64_t map_id = 0;
    std::int64_t map_generation = -1;
    std::uint64_t reference_kf_id = 0;
    Sophus::SE3f previous_tcw;
    std::uint64_t reference_kf_pose_hash = 0;
    std::uint64_t map_point_state_hash = 0;
    std::uint64_t static_support_hash = 0;
    std::uint64_t dynamic_support_hash = 0;
    std::uint64_t common_support_hash = 0;
};

struct ReplayTrackingTrace {
    bool motion_model_ran = false;
    std::uint64_t motion_previous_pose_hash = 0;
    std::uint64_t motion_static_initial_pose_hash = 0;
    std::uint64_t motion_static_support_hash = 0;
    std::uint64_t motion_static_support_count = 0;
    std::int64_t motion_static_inliers = -1;
    std::uint64_t motion_static_optimized_pose_hash = 0;
    std::uint64_t motion_dynamic_initial_pose_hash = 0;
    std::uint64_t motion_dynamic_support_hash = 0;
    std::uint64_t motion_dynamic_support_count = 0;
    std::int64_t motion_dynamic_inliers = -1;
    std::uint64_t motion_dynamic_optimized_pose_hash = 0;
    std::uint64_t motion_output_pose_hash = 0;
    bool local_map_ran = false;
    std::uint64_t local_map_entry_pose_hash = 0;
    std::uint64_t local_keyframe_sequence_hash = 0;
    std::uint64_t local_keyframe_count = 0;
    std::uint64_t local_candidate_state_hash = 0;
    std::uint64_t local_candidate_id_hash = 0;
    std::uint64_t local_candidate_position_hash = 0;
    std::uint64_t local_candidate_normal_hash = 0;
    std::uint64_t local_candidate_distance_hash = 0;
    std::uint64_t local_candidate_descriptor_hash = 0;
    std::uint64_t local_candidate_observation_hash = 0;
    std::uint64_t local_candidate_count = 0;
    std::uint64_t local_map_support_hash = 0;
    std::uint64_t local_map_support_count = 0;
    std::int64_t local_map_optimizer_inliers = -1;
    std::uint64_t local_map_optimized_pose_hash = 0;
};

struct ReplayExecutionState {
    bool valid = false;
    std::uint64_t frame_id = 0;
    std::uint64_t map_id = 0;
    std::int64_t map_generation = -1;
    std::uint64_t reference_kf_id = 0;
    bool is_keyframe = false;
    std::uint64_t current_gray_hash = 0;
    std::uint64_t current_depth_hash = 0;
    std::uint64_t current_static_mask_hash = 0;
    std::uint64_t current_descriptors_hash = 0;
    std::uint64_t feature_count = 0;
    ReplayTrackingTrace tracking;
    Sophus::SE3f current_tcw;
    std::uint64_t current_pose_hash = 0;
    std::uint64_t map_point_count = 0;
    std::uint64_t keyframe_count = 0;
    std::uint64_t map_point_hash = 0;
    std::uint64_t keyframe_hash = 0;
    std::uint64_t map_fingerprint = 0;
    ReplayFreezeState freeze;
};

Sophus::SE3f canonicalPreviousTcw(
    const Sophus::SE3f& frame_from_reference,
    const Sophus::SE3f& reference_tcw);

std::uint64_t hashPose(const Sophus::SE3f& pose);
std::uint64_t hashCvMat(const cv::Mat& matrix);
std::uint64_t hashMapPointStates(
    const std::vector<ReplayMapPointState>& states);
std::uint64_t hashMapPointTrackingStates(
    const std::vector<ReplayMapPointTrackingState>& states);
ReplayMapPointTrackingDigest digestMapPointTrackingStates(
    const std::vector<ReplayMapPointTrackingState>& states);
std::uint64_t hashKeyFrameStates(
    const std::vector<ReplayKeyFrameState>& states);
std::uint64_t hashIdSequence(
    const std::vector<std::uint64_t>& ids);
std::uint64_t hashExecutionMapState(
    std::uint64_t map_id,
    std::int64_t map_generation,
    std::uint64_t map_point_hash,
    std::uint64_t keyframe_hash);
std::uint64_t hashSupportObservations(
    const std::vector<ReplaySupportObservation>& observations);
std::string hashToHex(std::uint64_t hash);

ReplayStateIdentity buildReplayStateIdentity(
    std::uint64_t previous_frame_id,
    std::uint64_t map_id,
    std::int64_t map_generation,
    std::uint64_t reference_kf_id,
    const Sophus::SE3f& previous_tcw,
    const Sophus::SE3f& reference_kf_tcw,
    const std::vector<ReplayMapPointState>& map_points,
    const std::vector<ReplaySupportObservation>& static_support,
    const std::vector<ReplaySupportObservation>& dynamic_support,
    const std::vector<ReplaySupportObservation>& common_support);

}  // namespace Motion3D
