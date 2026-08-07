#include "Motion3D/FrozenReplayPacket.h"

#include "Frame.h"
#include "MapPoint.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Pinhole.h"

#include <opencv2/core/persistence.hpp>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>

namespace Motion3D {
namespace {

using ORB_SLAM3::Frame;
using ORB_SLAM3::MapPoint;

std::string uint64ToString(std::uint64_t value) {
    return std::to_string(value);
}

bool stringToUint64(
    const cv::FileNode& node,
    std::uint64_t& value) {
    if (!node.isString()) {
        return false;
    }
    try {
        std::size_t consumed = 0;
        value = std::stoull(static_cast<std::string>(node), &consumed);
        return consumed == static_cast<std::string>(node).size();
    } catch (const std::exception&) {
        return false;
    }
}

void writePose(
    cv::FileStorage& storage,
    const std::string& name,
    const Sophus::SE3f& pose) {
    const Eigen::Quaternionf rotation = pose.unit_quaternion();
    storage << name << "["
            << pose.translation().x()
            << pose.translation().y()
            << pose.translation().z()
            << rotation.x()
            << rotation.y()
            << rotation.z()
            << rotation.w()
            << "]";
}

bool readPose(const cv::FileNode& node, Sophus::SE3f& pose) {
    if (!node.isSeq() || node.size() != 7) {
        return false;
    }
    std::vector<float> values;
    node >> values;
    if (values.size() != 7 ||
        !std::all_of(
            values.begin(), values.end(),
            [](float value) { return std::isfinite(value); })) {
        return false;
    }
    const Eigen::Quaternionf rotation(
        values[6], values[3], values[4], values[5]);
    const float squaredNorm = rotation.squaredNorm();
    if (squaredNorm < 1e-8f ||
        std::abs(squaredNorm - 1.0f) >=
            Sophus::Constants<float>::epsilon()) {
        return false;
    }
    pose = Sophus::SE3f();
    float* raw = pose.data();
    raw[0] = values[3];
    raw[1] = values[4];
    raw[2] = values[5];
    raw[3] = values[6];
    raw[4] = values[0];
    raw[5] = values[1];
    raw[6] = values[2];
    return true;
}

void writeMatrix6(
    cv::FileStorage& storage,
    const std::string& name,
    const Eigen::Matrix<float, 6, 6>& matrix) {
    storage << name << "[";
    for (int row = 0; row < 6; ++row) {
        for (int column = 0; column < 6; ++column) {
            storage << matrix(row, column);
        }
    }
    storage << "]";
}

bool readMatrix6(
    const cv::FileNode& node,
    Eigen::Matrix<float, 6, 6>& matrix) {
    if (!node.isSeq() || node.size() != 36) {
        return false;
    }
    std::vector<float> values;
    node >> values;
    if (values.size() != 36 ||
        !std::all_of(
            values.begin(), values.end(),
            [](float value) { return std::isfinite(value); })) {
        return false;
    }
    for (int row = 0; row < 6; ++row) {
        for (int column = 0; column < 6; ++column) {
            matrix(row, column) = values[6 * row + column];
        }
    }
    return true;
}

void writeKeypoints(
    cv::FileStorage& storage,
    const std::string& name,
    const std::vector<cv::KeyPoint>& keypoints) {
    storage << name << "[";
    for (const cv::KeyPoint& keypoint : keypoints) {
        storage << "{:"
                << "x" << keypoint.pt.x
                << "y" << keypoint.pt.y
                << "size" << keypoint.size
                << "angle" << keypoint.angle
                << "response" << keypoint.response
                << "octave" << keypoint.octave
                << "class_id" << keypoint.class_id
                << "}";
    }
    storage << "]";
}

bool readKeypoints(
    const cv::FileNode& node,
    std::vector<cv::KeyPoint>& keypoints) {
    if (!node.isSeq()) {
        return false;
    }
    keypoints.clear();
    keypoints.reserve(node.size());
    for (const cv::FileNode& item : node) {
        if (!item.isMap()) {
            return false;
        }
        cv::KeyPoint keypoint;
        item["x"] >> keypoint.pt.x;
        item["y"] >> keypoint.pt.y;
        item["size"] >> keypoint.size;
        item["angle"] >> keypoint.angle;
        item["response"] >> keypoint.response;
        item["octave"] >> keypoint.octave;
        item["class_id"] >> keypoint.class_id;
        if (!std::isfinite(keypoint.pt.x) ||
            !std::isfinite(keypoint.pt.y) ||
            keypoint.octave < 0) {
            return false;
        }
        keypoints.push_back(keypoint);
    }
    return true;
}

void writeDirectResult(
    cv::FileStorage& storage,
    const std::string& name,
    const DirectRgbdPoseValidator::Result& result) {
    storage << name << "{"
            << "valid" << static_cast<int>(result.valid)
            << "common_support" << result.common_support
            << "static_depth_score" << result.static_depth_score
            << "dynamic_depth_score" << result.dynamic_depth_score
            << "static_photometric_score"
            << result.static_photometric_score
            << "dynamic_photometric_score"
            << result.dynamic_photometric_score
            << "static_combined_score" << result.static_combined_score
            << "dynamic_combined_score" << result.dynamic_combined_score
            << "}";
}

bool readDirectResult(
    const cv::FileNode& node,
    DirectRgbdPoseValidator::Result& result) {
    if (!node.isMap()) {
        return false;
    }
    int valid = 0;
    node["valid"] >> valid;
    node["common_support"] >> result.common_support;
    node["static_depth_score"] >> result.static_depth_score;
    node["dynamic_depth_score"] >> result.dynamic_depth_score;
    node["static_photometric_score"] >>
        result.static_photometric_score;
    node["dynamic_photometric_score"] >>
        result.dynamic_photometric_score;
    node["static_combined_score"] >> result.static_combined_score;
    node["dynamic_combined_score"] >> result.dynamic_combined_score;
    result.valid = valid != 0;
    return true;
}

void writeBranch(
    cv::FileStorage& storage,
    const std::string& name,
    const FrozenReplayBranchResult& branch) {
    storage << name << "{"
            << "matches" << branch.matches
            << "inliers" << branch.inliers;
    writePose(storage, "optimized_tcw", branch.optimized_tcw);
    storage << "}";
}

bool readBranch(
    const cv::FileNode& node,
    FrozenReplayBranchResult& branch) {
    if (!node.isMap()) {
        return false;
    }
    node["matches"] >> branch.matches;
    node["inliers"] >> branch.inliers;
    return readPose(node["optimized_tcw"], branch.optimized_tcw);
}

void writeStateIdentity(
    cv::FileStorage& storage,
    const ReplayStateIdentity& identity) {
    storage << "state_identity" << "{"
            << "valid" << static_cast<int>(identity.valid)
            << "contract_version" << identity.contract_version
            << "hash_algorithm" << identity.hash_algorithm
            << "previous_frame_id"
            << uint64ToString(identity.previous_frame_id)
            << "map_id" << uint64ToString(identity.map_id)
            << "map_generation"
            << static_cast<double>(identity.map_generation)
            << "reference_kf_id"
            << uint64ToString(identity.reference_kf_id);
    writePose(storage, "previous_tcw", identity.previous_tcw);
    storage
        << "reference_kf_pose_hash"
        << uint64ToString(identity.reference_kf_pose_hash)
        << "map_point_state_hash"
        << uint64ToString(identity.map_point_state_hash)
        << "static_support_hash"
        << uint64ToString(identity.static_support_hash)
        << "dynamic_support_hash"
        << uint64ToString(identity.dynamic_support_hash)
        << "common_support_hash"
        << uint64ToString(identity.common_support_hash)
        << "}";
}

bool readStateIdentity(
    const cv::FileNode& node,
    ReplayStateIdentity& identity) {
    if (!node.isMap()) {
        return false;
    }
    int valid = 0;
    double generation = -1.0;
    node["valid"] >> valid;
    node["contract_version"] >> identity.contract_version;
    node["hash_algorithm"] >> identity.hash_algorithm;
    node["map_generation"] >> generation;
    identity.valid = valid != 0;
    identity.map_generation = static_cast<std::int64_t>(generation);
    return stringToUint64(
               node["previous_frame_id"], identity.previous_frame_id) &&
        stringToUint64(node["map_id"], identity.map_id) &&
        stringToUint64(node["reference_kf_id"], identity.reference_kf_id) &&
        readPose(node["previous_tcw"], identity.previous_tcw) &&
        stringToUint64(
            node["reference_kf_pose_hash"],
            identity.reference_kf_pose_hash) &&
        stringToUint64(
            node["map_point_state_hash"],
            identity.map_point_state_hash) &&
        stringToUint64(
            node["static_support_hash"], identity.static_support_hash) &&
        stringToUint64(
            node["dynamic_support_hash"], identity.dynamic_support_hash) &&
        stringToUint64(
            node["common_support_hash"], identity.common_support_hash);
}

void writeFreezeState(
    cv::FileStorage& storage,
    const ReplayFreezeState& freeze) {
    storage << "freeze" << "{"
            << "protocol" << freeze.protocol
            << "requested" << static_cast<int>(freeze.requested)
            << "local_mapping_idle_ack"
            << static_cast<int>(freeze.local_mapping_idle_ack)
            << "local_mapping_stopped_ack"
            << static_cast<int>(freeze.local_mapping_stopped_ack)
            << "loop_closing_idle_ack"
            << static_cast<int>(freeze.loop_closing_idle_ack)
            << "loop_closing_stopped_ack"
            << static_cast<int>(freeze.loop_closing_stopped_ack)
            << "gba_stopped_ack"
            << static_cast<int>(freeze.gba_stopped_ack)
            << "epoch" << uint64ToString(freeze.epoch)
            << "}";
}

bool readFreezeState(
    const cv::FileNode& node,
    ReplayFreezeState& freeze) {
    if (!node.isMap()) {
        return false;
    }
    int requested = 0;
    int localIdle = 0;
    int localStopped = 0;
    int loopIdle = 0;
    int loopStopped = 0;
    int gbaStopped = 0;
    node["protocol"] >> freeze.protocol;
    node["requested"] >> requested;
    node["local_mapping_idle_ack"] >> localIdle;
    node["local_mapping_stopped_ack"] >> localStopped;
    node["loop_closing_idle_ack"] >> loopIdle;
    node["loop_closing_stopped_ack"] >> loopStopped;
    node["gba_stopped_ack"] >> gbaStopped;
    freeze.requested = requested != 0;
    freeze.local_mapping_idle_ack = localIdle != 0;
    freeze.local_mapping_stopped_ack = localStopped != 0;
    freeze.loop_closing_idle_ack = loopIdle != 0;
    freeze.loop_closing_stopped_ack = loopStopped != 0;
    freeze.gba_stopped_ack = gbaStopped != 0;
    return stringToUint64(node["epoch"], freeze.epoch);
}

std::vector<ReplaySupportObservation> supportObservations(
    const Frame& frame,
    const std::vector<MapPoint*>& matches) {
    std::vector<ReplaySupportObservation> support;
    const std::size_t count = std::min(
        matches.size(), static_cast<std::size_t>(frame.N));
    for (std::size_t index = 0; index < count; ++index) {
        MapPoint* point = matches[index];
        if (!point) {
            continue;
        }
        ReplaySupportObservation observation;
        observation.feature_index = index;
        observation.map_point_id = point->mnId;
        observation.world_position = point->GetWorldPos();
        observation.keypoint_x = frame.mvKeysUn[index].pt.x;
        observation.keypoint_y = frame.mvKeysUn[index].pt.y;
        observation.right_coordinate = frame.mvuRight[index];
        observation.octave = frame.mvKeysUn[index].octave;
        support.push_back(observation);
    }
    return support;
}

void assignGrid(Frame& frame) {
    for (int x = 0; x < FRAME_GRID_COLS; ++x) {
        for (int y = 0; y < FRAME_GRID_ROWS; ++y) {
            frame.mGrid[x][y].clear();
        }
    }
    for (std::size_t index = 0; index < frame.mvKeysUn.size(); ++index) {
        const cv::KeyPoint& keypoint = frame.mvKeysUn[index];
        const int gridX = static_cast<int>(std::lround(
            (keypoint.pt.x - Frame::mnMinX) *
            Frame::mfGridElementWidthInv));
        const int gridY = static_cast<int>(std::lround(
            (keypoint.pt.y - Frame::mnMinY) *
            Frame::mfGridElementHeightInv));
        if (gridX >= 0 && gridX < FRAME_GRID_COLS &&
            gridY >= 0 && gridY < FRAME_GRID_ROWS) {
            frame.mGrid[gridX][gridY].push_back(index);
        }
    }
}

std::unique_ptr<Frame> makeCurrentFrame(
    const FrozenReplayPacket& packet,
    ORB_SLAM3::Pinhole* camera,
    const Sophus::SE3f& initialPose) {
    std::unique_ptr<Frame> frame(new Frame());
    frame->N = packet.feature_count;
    frame->Nleft = -1;
    frame->Nright = -1;
    frame->mnScaleLevels = packet.scale_levels;
    frame->mfScaleFactor = packet.scale_factor;
    frame->mvScaleFactors = packet.scale_factors;
    frame->mvInvLevelSigma2 = packet.inverse_level_sigma2;
    frame->mvKeys = packet.current_keypoints;
    frame->mvKeysUn = packet.current_keypoints;
    frame->mvuRight = packet.current_right_coordinates;
    frame->mDescriptors = packet.current_descriptors.clone();
    frame->mvpMapPoints.assign(frame->N, nullptr);
    frame->mvbOutlier.assign(frame->N, false);
    frame->mpCamera = camera;
    frame->mpCamera2 = nullptr;
    frame->mb = packet.baseline;
    frame->mbf = packet.baseline_fx;
    frame->SetPose(initialPose);
    assignGrid(*frame);
    return frame;
}

bool scoreCommonSupport(
    const Frame& frame,
    const std::vector<MapPoint*>& staticMatches,
    const std::vector<MapPoint*>& dynamicMatches,
    const Sophus::SE3f& staticPose,
    const Sophus::SE3f& dynamicPose,
    int& commonSupport,
    float& staticScore,
    float& dynamicScore,
    std::vector<ReplaySupportObservation>& observations) {
    commonSupport = 0;
    staticScore = -1.0f;
    dynamicScore = -1.0f;
    observations.clear();
    if (staticMatches.size() != dynamicMatches.size()) {
        return false;
    }
    double staticCost = 0.0;
    double dynamicCost = 0.0;
    const std::size_t count = std::min(
        staticMatches.size(), static_cast<std::size_t>(frame.N));
    for (std::size_t index = 0; index < count; ++index) {
        MapPoint* point = staticMatches[index];
        if (!point || point != dynamicMatches[index]) {
            continue;
        }
        const Eigen::Vector3f world = point->GetWorldPos();
        const auto scorePose = [&](const Sophus::SE3f& pose, double& cost) {
            const Eigen::Vector3f cameraPoint = pose * world;
            if (!cameraPoint.allFinite() || cameraPoint.z() <= 1e-6f) {
                return false;
            }
            const Eigen::Vector2f projection =
                frame.mpCamera->project(cameraPoint);
            if (!projection.allFinite() ||
                projection.x() < Frame::mnMinX ||
                projection.x() > Frame::mnMaxX ||
                projection.y() < Frame::mnMinY ||
                projection.y() > Frame::mnMaxY) {
                return false;
            }
            const cv::KeyPoint& keypoint = frame.mvKeysUn[index];
            const double dx = projection.x() - keypoint.pt.x;
            const double dy = projection.y() - keypoint.pt.y;
            double squaredError = dx * dx + dy * dy;
            double cutoff = 5.991;
            if (frame.mvuRight[index] >= 0.0f) {
                const double projectedRight =
                    projection.x() - frame.mbf / cameraPoint.z();
                const double rightError =
                    projectedRight - frame.mvuRight[index];
                squaredError += rightError * rightError;
                cutoff = 7.815;
            }
            const int octave = keypoint.octave;
            if (octave < 0 ||
                octave >= static_cast<int>(
                    frame.mvInvLevelSigma2.size())) {
                return false;
            }
            const double chiSquared =
                squaredError * frame.mvInvLevelSigma2[octave];
            cost = chiSquared <= cutoff
                ? chiSquared
                : 2.0 * std::sqrt(cutoff * chiSquared) - cutoff;
            return std::isfinite(cost);
        };
        double staticPointCost = 0.0;
        double dynamicPointCost = 0.0;
        if (!scorePose(staticPose, staticPointCost) ||
            !scorePose(dynamicPose, dynamicPointCost)) {
            continue;
        }
        staticCost += staticPointCost;
        dynamicCost += dynamicPointCost;
        ReplaySupportObservation observation;
        observation.feature_index = index;
        observation.map_point_id = point->mnId;
        observation.world_position = world;
        observation.keypoint_x = frame.mvKeysUn[index].pt.x;
        observation.keypoint_y = frame.mvKeysUn[index].pt.y;
        observation.right_coordinate = frame.mvuRight[index];
        observation.octave = frame.mvKeysUn[index].octave;
        observations.push_back(observation);
        ++commonSupport;
    }
    if (commonSupport > 0) {
        staticScore = static_cast<float>(staticCost / commonSupport);
        dynamicScore = static_cast<float>(dynamicCost / commonSupport);
    }
    return true;
}

bool validatePacket(const FrozenReplayPacket& packet, std::string& error) {
    if (!packet.valid ||
        packet.packet_version != kFrozenReplayPacketVersion ||
        packet.hash_algorithm != kReplayStateHashAlgorithm ||
        packet.freeze.protocol != kReplayFreezeProtocolVersion ||
        !packet.freeze.complete() ||
        packet.freeze.epoch == 0) {
        error = "Unsupported or invalid replay packet contract";
        return false;
    }
    if (!packet.dynamic_pose_information.allFinite() ||
        !std::isfinite(packet.dynamic_information_scale) ||
        packet.dynamic_information_scale < 0.0f ||
        packet.dynamic_information_scale > 1.0f ||
        !std::isfinite(packet.shadow_translation_blend) ||
        packet.shadow_translation_blend < 0.0f ||
        packet.shadow_translation_blend > 1.0f ||
        !std::isfinite(packet.posterior_min_score_improvement) ||
        packet.posterior_min_score_improvement < 0.0f) {
        error = "Replay packet has invalid dynamic-prior fields";
        return false;
    }
    if (packet.posterior_evaluated &&
        (!packet.local_map_snapshot_valid ||
         !packet.posterior_static_initial_tcw.matrix().allFinite() ||
         !packet.posterior_dynamic_prior_tcw.matrix().allFinite() ||
         !packet.posterior_dynamic_information.allFinite())) {
        error = "Replay packet has invalid posterior inputs";
        return false;
    }
    if (packet.feature_count <= 0 ||
        packet.previous_feature_count <= 0 ||
        packet.current_keypoints.size() !=
            static_cast<std::size_t>(packet.feature_count) ||
        packet.previous_keypoints.size() !=
            static_cast<std::size_t>(packet.previous_feature_count) ||
        packet.current_right_coordinates.size() !=
            static_cast<std::size_t>(packet.feature_count) ||
        packet.previous_map_point_ids.size() !=
            static_cast<std::size_t>(packet.previous_feature_count) ||
        packet.previous_outliers.size() !=
            static_cast<std::size_t>(packet.previous_feature_count) ||
        packet.current_descriptors.rows != packet.feature_count ||
        packet.current_descriptors.type() != CV_8UC1) {
        error = "Replay packet feature arrays are inconsistent";
        return false;
    }
    if (packet.local_map_snapshot_valid) {
        if (packet.local_map_point_ids.size() !=
                static_cast<std::size_t>(packet.feature_count) ||
            packet.local_map_outliers.size() !=
                static_cast<std::size_t>(packet.feature_count)) {
            error = "Replay packet local-map support is inconsistent";
            return false;
        }
        std::unordered_map<std::uint64_t, const FrozenReplayMapPoint*>
            pointsById;
        for (const FrozenReplayMapPoint& point : packet.map_points)
            pointsById[point.id] = &point;
        std::vector<ReplaySupportObservation> support;
        for (std::size_t index = 0;
             index < packet.local_map_point_ids.size(); ++index) {
            const std::uint64_t id = packet.local_map_point_ids[index];
            if (id == kNoReplayMapPoint)
                continue;
            const auto found = pointsById.find(id);
            if (found == pointsById.end()) {
                error =
                    "Replay packet local-map support references missing MapPoint";
                return false;
            }
            ReplaySupportObservation observation;
            observation.feature_index = index;
            observation.map_point_id = id;
            observation.world_position =
                found->second->world_position;
            observation.keypoint_x =
                packet.current_keypoints[index].pt.x;
            observation.keypoint_y =
                packet.current_keypoints[index].pt.y;
            observation.right_coordinate =
                packet.current_right_coordinates[index];
            observation.octave =
                packet.current_keypoints[index].octave;
            support.push_back(observation);
        }
        if (support.empty() ||
            hashSupportObservations(support) !=
                packet.local_map_support_hash) {
            error = "Replay packet local-map support hash mismatch";
            return false;
        }
    }
    if (hashCvMat(packet.previous_gray) != packet.previous_gray_hash ||
        hashCvMat(packet.current_gray) != packet.current_gray_hash ||
        hashCvMat(packet.previous_depth) != packet.previous_depth_hash ||
        hashCvMat(packet.current_depth) != packet.current_depth_hash ||
        hashCvMat(packet.previous_static_mask) !=
            packet.previous_static_mask_hash ||
        hashCvMat(packet.current_static_mask) !=
            packet.current_static_mask_hash) {
        error = "Replay packet input asset hash mismatch";
        return false;
    }
    return true;
}

}  // namespace

namespace {

bool scoreEquivalent(
    float expected,
    float actual,
    float absoluteTolerance) {
    if (expected < 0.0f && actual < 0.0f) {
        return true;
    }
    if (!std::isfinite(expected) || !std::isfinite(actual)) {
        return false;
    }
    const float scale = std::max(std::abs(expected), std::abs(actual));
    return std::abs(expected - actual) <=
        absoluteTolerance +
            kReplayScoreRelativeTolerance * scale;
}

}  // namespace

bool replayScoreEquivalent(float expected, float actual) {
    return scoreEquivalent(
        expected, actual, kReplayScoreAbsoluteTolerance);
}

bool replayDirectScoreEquivalent(float expected, float actual) {
    return scoreEquivalent(
        expected, actual, kReplayDirectScoreAbsoluteTolerance);
}

bool replayPacketPastFreezeSettlingFrame(
    int frame_index,
    int freeze_map_after_frame) {
    return freeze_map_after_frame >= 0 &&
        frame_index > freeze_map_after_frame + kReplayFreezeSettlingFrames;
}

const char* replayGatePolicyName(ReplayGatePolicy policy) {
    switch (policy) {
        case ReplayGatePolicy::LegacyConjunction:
            return "legacy-conjunction";
        case ReplayGatePolicy::DirectCombined:
            return "direct-combined";
    }
    return "unknown";
}

bool parseReplayGatePolicy(
    const std::string& name, ReplayGatePolicy& policy) {
    if (name == "legacy-conjunction") {
        policy = ReplayGatePolicy::LegacyConjunction;
        return true;
    }
    if (name == "direct-combined") {
        policy = ReplayGatePolicy::DirectCombined;
        return true;
    }
    return false;
}

namespace {

bool replayInnovationIsBounded(
    const ReplayGateConfig& config,
    const Sophus::SE3f& static_tcw,
    const Sophus::SE3f& dynamic_tcw) {
    const Sophus::SE3f static_twc = static_tcw.inverse();
    const Sophus::SE3f dynamic_twc = dynamic_tcw.inverse();
    const float translation_innovation =
        (dynamic_twc.translation() - static_twc.translation()).norm();
    const float rotation_innovation =
        (static_twc.so3().inverse() * dynamic_twc.so3()).log().norm();
    return std::isfinite(translation_innovation) &&
        std::isfinite(rotation_innovation) &&
        translation_innovation >= config.min_translation_innovation &&
        translation_innovation <= config.max_translation_innovation &&
        rotation_innovation <= config.max_rotation_innovation;
}

}  // namespace

bool evaluateConsensusGate(
    const ReplayGateConfig& config,
    int static_inliers,
    int dynamic_inliers,
    const Sophus::SE3f& static_tcw,
    const Sophus::SE3f& dynamic_tcw) {
    if (config.bypass_reliability_gate) {
        return true;
    }
    const int minimumGain = std::max(
        config.min_inlier_gain,
        static_cast<int>(std::ceil(
            config.min_inlier_gain_ratio * static_inliers)));
    const bool recoversTracking =
        static_inliers < 10 && dynamic_inliers >= 10;
    const bool weakStaticSupport =
        static_inliers < config.max_static_inliers;
    const bool improvesStaticSupport =
        dynamic_inliers >= static_inliers + minimumGain;
    return replayInnovationIsBounded(
               config, static_tcw, dynamic_tcw) &&
        weakStaticSupport &&
        (recoversTracking || improvesStaticSupport);
}

bool evaluateReplayGatePolicy(
    const ReplayGateConfig& config,
    const Sophus::SE3f& static_tcw,
    const Sophus::SE3f& dynamic_tcw,
    const DirectRgbdPoseValidator::Result& direct,
    bool legacy_consensus_pass,
    bool legacy_direct_pass,
    bool legacy_common_support_pass) {
    if (config.policy == ReplayGatePolicy::LegacyConjunction) {
        return legacy_consensus_pass &&
            legacy_direct_pass &&
            legacy_common_support_pass;
    }
    if (config.policy != ReplayGatePolicy::DirectCombined ||
        config.bypass_reliability_gate ||
        !direct.valid ||
        !std::isfinite(direct.static_combined_score) ||
        !std::isfinite(direct.dynamic_combined_score) ||
        direct.static_combined_score < 0.0f ||
        direct.dynamic_combined_score < 0.0f ||
        direct.dynamic_combined_score >= direct.static_combined_score) {
        return false;
    }
    return replayInnovationIsBounded(
        config, static_tcw, dynamic_tcw);
}

void projectPosePriorToTranslationSubspace(
    const Sophus::SE3f& reference_tcw,
    Sophus::SE3f& prior_tcw,
    Eigen::Matrix<float, 6, 6>& information) {
    Eigen::Matrix<float, 6, 1> increment =
        (prior_tcw * reference_tcw.inverse()).log();
    increment.tail<3>().setZero();
    prior_tcw = Sophus::SE3f::exp(increment) * reference_tcw;

    Eigen::Matrix<float, 6, 6> projector =
        Eigen::Matrix<float, 6, 6>::Zero();
    projector.topLeftCorner<3, 3>().setIdentity();
    information = projector * information * projector;
    information = 0.5f * (information + information.transpose());
}

Sophus::SE3f projectCameraInterventionToTranslation(
    const Sophus::SE3f& static_tcw,
    const Sophus::SE3f& dynamic_tcw) {
    const Sophus::SE3f static_twc = static_tcw.inverse();
    const Sophus::SE3f dynamic_twc = dynamic_tcw.inverse();
    return Sophus::SE3f(
        static_twc.so3(), dynamic_twc.translation()).inverse();
}

Sophus::SE3f reuseShadowCameraTranslation(
    const Sophus::SE3f& live_tcw,
    const Sophus::SE3f& shadow_tcw,
    float translation_blend) {
    const float blend =
        std::min(1.0f, std::max(0.0f, translation_blend));
    const Sophus::SE3f live_twc = live_tcw.inverse();
    const Sophus::SE3f shadow_twc = shadow_tcw.inverse();
    const Eigen::Vector3f blended_translation =
        live_twc.translation() +
        blend * (shadow_twc.translation() - live_twc.translation());
    return Sophus::SE3f(
        live_twc.so3(), blended_translation).inverse();
}

Eigen::Matrix<float, 6, 6> isotropicTranslationInformation(
    const Eigen::Matrix<float, 6, 6>& information) {
    Eigen::Matrix<float, 6, 6> result =
        Eigen::Matrix<float, 6, 6>::Zero();
    const float meanInformation = std::max(
        0.0f, information.block<3, 3>(0, 0).trace() / 3.0f);
    result.block<3, 3>(0, 0) =
        meanInformation * Eigen::Matrix3f::Identity();
    return result;
}

const char* translationLeverageModeName(TranslationLeverageMode mode) {
    switch (mode) {
        case TranslationLeverageMode::CapOnly:
            return "cap-only";
        case TranslationLeverageMode::NormalizeToTarget:
            return "normalize-to-target";
    }
    return "invalid";
}

bool parseTranslationLeverageMode(
    const std::string& name, TranslationLeverageMode& mode) {
    if (name == "cap-only") {
        mode = TranslationLeverageMode::CapOnly;
        return true;
    }
    if (name == "normalize-to-target") {
        mode = TranslationLeverageMode::NormalizeToTarget;
        return true;
    }
    return false;
}

Eigen::Matrix<float, 6, 6> leverageCalibratedTranslationInformation(
    const Eigen::Matrix<float, 6, 6>& static_information,
    const Eigen::Matrix<float, 6, 6>& dynamic_information,
    float leverage_value,
    TranslationLeverageMode mode,
    TranslationLeverageDiagnostics* diagnostics) {
    if (diagnostics)
        *diagnostics = TranslationLeverageDiagnostics();
    Eigen::Matrix<float, 6, 6> result =
        Eigen::Matrix<float, 6, 6>::Zero();
    if (!static_information.allFinite() ||
        !dynamic_information.allFinite() ||
        !std::isfinite(leverage_value) ||
        leverage_value <= 0.0f ||
        (mode != TranslationLeverageMode::CapOnly &&
         mode != TranslationLeverageMode::NormalizeToTarget)) {
        return result;
    }

    using Matrix3f = Eigen::Matrix3f;
    using Matrix6f = Eigen::Matrix<float, 6, 6>;
    const Matrix6f static_symmetric =
        0.5f * (static_information + static_information.transpose());
    const Matrix3f rotation_block =
        static_symmetric.block<3, 3>(3, 3);
    Eigen::SelfAdjointEigenSolver<Matrix3f> rotation_solver(
        rotation_block);
    if (rotation_solver.info() != Eigen::Success)
        return result;
    const Eigen::Vector3f rotation_values =
        rotation_solver.eigenvalues().cwiseMax(0.0f);
    const float rotation_floor = 1e-6f * std::max(
        1.0f, rotation_values.maxCoeff());
    Eigen::Vector3f inverse_rotation_values =
        Eigen::Vector3f::Zero();
    for (int index = 0; index < 3; ++index) {
        if (rotation_values[index] > rotation_floor)
            inverse_rotation_values[index] =
                1.0f / rotation_values[index];
    }
    const Matrix3f rotation_pseudoinverse =
        rotation_solver.eigenvectors() *
        inverse_rotation_values.asDiagonal() *
        rotation_solver.eigenvectors().transpose();
    const Matrix3f translation_rotation =
        static_symmetric.block<3, 3>(0, 3);
    const Matrix3f static_schur = 0.5f * (
        static_symmetric.block<3, 3>(0, 0) -
        translation_rotation * rotation_pseudoinverse *
            translation_rotation.transpose() +
        (static_symmetric.block<3, 3>(0, 0) -
         translation_rotation * rotation_pseudoinverse *
            translation_rotation.transpose()).transpose());
    const Matrix3f dynamic_block = 0.5f * (
        dynamic_information.block<3, 3>(0, 0) +
        dynamic_information.block<3, 3>(0, 0).transpose());

    Eigen::SelfAdjointEigenSolver<Matrix3f> static_solver(static_schur);
    Eigen::SelfAdjointEigenSolver<Matrix3f> dynamic_solver(dynamic_block);
    if (static_solver.info() != Eigen::Success ||
        dynamic_solver.info() != Eigen::Success) {
        return result;
    }

    const Eigen::Vector3f static_values =
        static_solver.eigenvalues().cwiseMax(0.0f);
    const Eigen::Vector3f dynamic_values =
        dynamic_solver.eigenvalues().cwiseMax(0.0f);
    const float static_scale =
        std::max(1.0f, static_values.maxCoeff());
    const float static_floor = 1e-6f * static_scale;
    const Matrix3f dynamic_psd =
        dynamic_solver.eigenvectors() *
        dynamic_values.asDiagonal() *
        dynamic_solver.eigenvectors().transpose();
    Eigen::Vector3f square_root_values = Eigen::Vector3f::Zero();
    Eigen::Vector3f inverse_square_root_values =
        Eigen::Vector3f::Zero();
    Eigen::Vector3f support_values = Eigen::Vector3f::Zero();
    for (int index = 0; index < 3; ++index) {
        if (static_values[index] > static_floor) {
            square_root_values[index] =
                std::sqrt(static_values[index]);
            inverse_square_root_values[index] =
                1.0f / square_root_values[index];
            support_values[index] = 1.0f;
        }
    }
    const Matrix3f square_root =
        static_solver.eigenvectors() *
        square_root_values.asDiagonal() *
        static_solver.eigenvectors().transpose();
    const Matrix3f inverse_square_root =
        static_solver.eigenvectors() *
        inverse_square_root_values.asDiagonal() *
        static_solver.eigenvectors().transpose();
    const Matrix3f support_projector =
        static_solver.eigenvectors() *
        support_values.asDiagonal() *
        static_solver.eigenvectors().transpose();
    const Matrix3f supported_dynamic =
        support_projector * dynamic_psd * support_projector;

    const Matrix3f whitened = 0.5f * (
        inverse_square_root * supported_dynamic * inverse_square_root +
        (inverse_square_root * supported_dynamic * inverse_square_root)
            .transpose());
    Eigen::SelfAdjointEigenSolver<Matrix3f> leverage_solver(whitened);
    if (leverage_solver.info() != Eigen::Success)
        return result;
    const Eigen::Vector3f raw_leverage_values =
        leverage_solver.eigenvalues().cwiseMax(0.0f);
    Eigen::Vector3f bounded_values = Eigen::Vector3f::Zero();
    float normalization_scale = 1.0f;
    if (mode == TranslationLeverageMode::CapOnly) {
        bounded_values =
            raw_leverage_values.cwiseMin(leverage_value);
    } else {
        const float raw_max = raw_leverage_values.maxCoeff();
        const float leverage_floor =
            std::numeric_limits<float>::epsilon() *
            std::max(1.0f, leverage_value);
        if (raw_max > leverage_floor) {
            normalization_scale = leverage_value / raw_max;
            bounded_values =
                normalization_scale * raw_leverage_values;
        } else {
            normalization_scale = 0.0f;
        }
    }
    const Matrix3f bounded = square_root *
        leverage_solver.eigenvectors() *
        bounded_values.asDiagonal() *
        leverage_solver.eigenvectors().transpose() *
        square_root;
    result.block<3, 3>(0, 0) = 0.5f * (bounded + bounded.transpose());
    if (diagnostics) {
        const float unsupported_norm =
            (dynamic_psd - supported_dynamic).norm();
        const float unsupported_tolerance =
            1e-5f * std::max(1.0f, dynamic_psd.norm());
        diagnostics->valid = true;
        diagnostics->static_translation_trace =
            static_values.sum();
        diagnostics->dynamic_translation_trace_raw =
            dynamic_values.sum();
        diagnostics->dynamic_translation_trace_bounded =
            result.block<3, 3>(0, 0).trace();
        diagnostics->max_generalized_leverage_raw =
            unsupported_norm > unsupported_tolerance
            ? std::numeric_limits<float>::infinity()
            : raw_leverage_values.maxCoeff();
        diagnostics->max_generalized_leverage_bounded =
            bounded_values.maxCoeff();
        diagnostics->normalization_scale = normalization_scale;
    }
    return result;
}

Eigen::Matrix<float, 6, 6> leverageBoundedTranslationInformation(
    const Eigen::Matrix<float, 6, 6>& static_information,
    const Eigen::Matrix<float, 6, 6>& dynamic_information,
    float max_generalized_leverage,
    TranslationLeverageDiagnostics* diagnostics) {
    return leverageCalibratedTranslationInformation(
        static_information, dynamic_information,
        max_generalized_leverage, TranslationLeverageMode::CapOnly,
        diagnostics);
}

Sophus::SE3f updateCameraVelocity(
    const Sophus::SE3f& incoming_velocity,
    const Sophus::SE3f& previous_tcw,
    const Sophus::SE3f& current_tcw,
    bool velocity_neutral) {
    if (velocity_neutral)
        return incoming_velocity;
    return current_tcw * previous_tcw.inverse();
}

bool posteriorLocalMapGate(
    int static_inliers,
    int dynamic_inliers,
    float static_score,
    float dynamic_score,
    float minimum_score_improvement) {
    if (static_inliers < 0 || dynamic_inliers < 0 ||
        !std::isfinite(static_score) || !std::isfinite(dynamic_score) ||
        !std::isfinite(minimum_score_improvement) ||
        static_score < 0.0f || dynamic_score < 0.0f ||
        minimum_score_improvement < 0.0f) {
        return false;
    }
    const float tolerance =
        1e-4f + 1e-4f * std::max(
            std::abs(static_score), std::abs(dynamic_score));
    if (dynamic_inliers > static_inliers)
        return dynamic_score <= static_score + tolerance;
    return dynamic_inliers == static_inliers &&
        dynamic_score + minimum_score_improvement < static_score;
}

bool writeFrozenReplayPacket(
    const FrozenReplayPacket& packet,
    const std::string& path,
    std::string* error) {
    cv::FileStorage storage(path, cv::FileStorage::WRITE);
    if (!storage.isOpened()) {
        if (error) *error = "Cannot open replay packet for writing: " + path;
        return false;
    }
    storage
        << "valid" << static_cast<int>(packet.valid)
        << "packet_version" << packet.packet_version
        << "hash_algorithm" << packet.hash_algorithm
        << "branch_type" << packet.branch_type
        << "previous_image_name" << packet.previous_image_name
        << "current_image_name" << packet.current_image_name;
    writeStateIdentity(storage, packet.state_identity);
    writeFreezeState(storage, packet.freeze);
    storage
        << "previous_timestamp" << packet.previous_timestamp
        << "current_timestamp" << packet.current_timestamp;
    writePose(storage, "previous_tcw", packet.previous_tcw);
    writePose(storage, "velocity", packet.velocity);
    writePose(storage, "static_initial_tcw", packet.static_initial_tcw);
    writePose(storage, "dynamic_initial_tcw", packet.dynamic_initial_tcw);
    writeMatrix6(
        storage, "dynamic_pose_information",
        packet.dynamic_pose_information);
    storage
        << "dynamic_information_scale"
        << packet.dynamic_information_scale
        << "dynamic_initialization_only"
        << static_cast<int>(packet.dynamic_initialization_only)
        << "search_threshold" << packet.search_threshold
        << "minimum_matches" << packet.minimum_matches
        << "feature_count" << packet.feature_count
        << "previous_feature_count" << packet.previous_feature_count
        << "scale_levels" << packet.scale_levels
        << "scale_factor" << packet.scale_factor
        << "baseline" << packet.baseline
        << "baseline_fx" << packet.baseline_fx
        << "fx" << packet.fx << "fy" << packet.fy
        << "cx" << packet.cx << "cy" << packet.cy
        << "min_x" << packet.min_x << "max_x" << packet.max_x
        << "min_y" << packet.min_y << "max_y" << packet.max_y
        << "grid_width_inverse" << packet.grid_width_inverse
        << "grid_height_inverse" << packet.grid_height_inverse
        << "scale_factors" << packet.scale_factors
        << "inverse_level_sigma2" << packet.inverse_level_sigma2;
    writeKeypoints(storage, "current_keypoints", packet.current_keypoints);
    writeKeypoints(storage, "previous_keypoints", packet.previous_keypoints);
    storage
        << "current_right_coordinates"
        << packet.current_right_coordinates
        << "current_descriptors" << packet.current_descriptors;
    storage << "previous_map_point_ids" << "[";
    for (std::uint64_t id : packet.previous_map_point_ids) {
        storage << uint64ToString(id);
    }
    storage << "]" << "previous_outliers" << "[";
    for (unsigned char outlier : packet.previous_outliers) {
        storage << static_cast<int>(outlier);
    }
    storage << "]" << "map_points" << "[";
    for (const FrozenReplayMapPoint& point : packet.map_points) {
        storage << "{"
                << "id" << uint64ToString(point.id)
                << "world_position" << "["
                << point.world_position.x()
                << point.world_position.y()
                << point.world_position.z()
                << "]"
                << "observations" << point.observations
                << "descriptor" << point.descriptor
                << "}";
    }
    storage
        << "]"
        << "local_map_snapshot_valid"
        << static_cast<int>(packet.local_map_snapshot_valid)
        << "local_map_point_ids" << "[";
    for (std::uint64_t id : packet.local_map_point_ids)
        storage << uint64ToString(id);
    storage
        << "]"
        << "local_map_outliers" << "[";
    for (unsigned char outlier : packet.local_map_outliers)
        storage << static_cast<int>(outlier);
    storage
        << "]"
        << "local_map_support_hash"
        << uint64ToString(packet.local_map_support_hash);
    storage
        << "previous_gray" << packet.previous_gray
        << "current_gray" << packet.current_gray
        << "previous_depth" << packet.previous_depth
        << "current_depth" << packet.current_depth
        << "previous_static_mask" << packet.previous_static_mask
        << "current_static_mask" << packet.current_static_mask
        << "previous_gray_hash"
        << uint64ToString(packet.previous_gray_hash)
        << "current_gray_hash"
        << uint64ToString(packet.current_gray_hash)
        << "previous_depth_hash"
        << uint64ToString(packet.previous_depth_hash)
        << "current_depth_hash"
        << uint64ToString(packet.current_depth_hash)
        << "previous_static_mask_hash"
        << uint64ToString(packet.previous_static_mask_hash)
        << "current_static_mask_hash"
        << uint64ToString(packet.current_static_mask_hash);
    storage << "gate" << "{:"
            << "policy" << replayGatePolicyName(packet.gate.policy)
            << "min_inlier_gain" << packet.gate.min_inlier_gain
            << "min_inlier_gain_ratio"
            << packet.gate.min_inlier_gain_ratio
            << "max_static_inliers" << packet.gate.max_static_inliers
            << "min_translation_innovation"
            << packet.gate.min_translation_innovation
            << "max_translation_innovation"
            << packet.gate.max_translation_innovation
            << "max_rotation_innovation"
            << packet.gate.max_rotation_innovation
            << "use_direct_validation"
            << static_cast<int>(packet.gate.use_direct_validation)
            << "direct_score_mode" << packet.gate.direct_score_mode
            << "require_common_support_improvement"
            << static_cast<int>(
                packet.gate.require_common_support_improvement)
            << "bypass_reliability_gate"
            << static_cast<int>(
                packet.gate.bypass_reliability_gate)
            << "}";
    storage << "direct_config" << "{:"
            << "grid_step" << packet.direct_config.grid_step
            << "min_common_support"
            << packet.direct_config.min_common_support
            << "min_depth" << packet.direct_config.min_depth
            << "max_depth" << packet.direct_config.max_depth
            << "depth_sigma_constant"
            << packet.direct_config.depth_sigma_constant
            << "depth_sigma_linear"
            << packet.direct_config.depth_sigma_linear
            << "photometric_sigma"
            << packet.direct_config.photometric_sigma
            << "depth_weight" << packet.direct_config.depth_weight
            << "photometric_weight"
            << packet.direct_config.photometric_weight
            << "}";
    writeBranch(storage, "expected_static", packet.expected_static);
    writeBranch(storage, "expected_dynamic", packet.expected_dynamic);
    storage
        << "expected_common_support" << packet.expected_common_support
        << "expected_static_common_score"
        << packet.expected_static_common_score
        << "expected_dynamic_common_score"
        << packet.expected_dynamic_common_score;
    writeDirectResult(storage, "expected_direct", packet.expected_direct);
    storage
        << "expected_consensus_pass"
        << static_cast<int>(packet.expected_consensus_pass)
        << "expected_direct_pass"
        << static_cast<int>(packet.expected_direct_pass)
        << "expected_common_support_pass"
        << static_cast<int>(packet.expected_common_support_pass)
        << "expected_would_use"
        << static_cast<int>(packet.expected_would_use)
        << "shadow_translation_blend"
        << packet.shadow_translation_blend
        << "posterior_min_score_improvement"
        << packet.posterior_min_score_improvement;
    writeBranch(
        storage, "expected_posterior_static",
        packet.expected_posterior_static);
    writeBranch(
        storage, "expected_posterior_dynamic",
        packet.expected_posterior_dynamic);
    storage
        << "expected_posterior_common_support"
        << packet.expected_posterior_common_support
        << "expected_posterior_static_score"
        << packet.expected_posterior_static_score
        << "expected_posterior_dynamic_score"
        << packet.expected_posterior_dynamic_score
        << "expected_posterior_valid"
        << static_cast<int>(packet.expected_posterior_valid)
        << "expected_posterior_pass"
        << static_cast<int>(packet.expected_posterior_pass);
    storage
        << "posterior_evaluated"
        << static_cast<int>(packet.posterior_evaluated);
    writePose(
        storage, "posterior_static_initial_tcw",
        packet.posterior_static_initial_tcw);
    writePose(
        storage, "posterior_dynamic_prior_tcw",
        packet.posterior_dynamic_prior_tcw);
    writeMatrix6(
        storage, "posterior_dynamic_information",
        packet.posterior_dynamic_information);
    storage.release();
    return true;
}

bool readFrozenReplayPacket(
    const std::string& path,
    FrozenReplayPacket& packet,
    std::string* error) {
    cv::FileStorage storage(path, cv::FileStorage::READ);
    if (!storage.isOpened()) {
        if (error) *error = "Cannot open replay packet: " + path;
        return false;
    }
    int valid = 0;
    storage["valid"] >> valid;
    packet.valid = valid != 0;
    storage["packet_version"] >> packet.packet_version;
    storage["hash_algorithm"] >> packet.hash_algorithm;
    storage["branch_type"] >> packet.branch_type;
    storage["previous_image_name"] >> packet.previous_image_name;
    storage["current_image_name"] >> packet.current_image_name;
    if (!readStateIdentity(
            storage["state_identity"], packet.state_identity) ||
        !readFreezeState(storage["freeze"], packet.freeze) ||
        !readPose(storage["previous_tcw"], packet.previous_tcw) ||
        !readPose(storage["velocity"], packet.velocity) ||
        !readPose(
            storage["static_initial_tcw"], packet.static_initial_tcw) ||
        !readPose(
            storage["dynamic_initial_tcw"], packet.dynamic_initial_tcw)) {
        if (error) *error = "Replay packet has invalid state or pose fields";
        return false;
    }
    int dynamicInitializationOnly = 0;
    if (!readMatrix6(
            storage["dynamic_pose_information"],
            packet.dynamic_pose_information)) {
        if (error) *error = "Replay packet has invalid dynamic information";
        return false;
    }
    storage["dynamic_information_scale"] >>
        packet.dynamic_information_scale;
    storage["dynamic_initialization_only"] >>
        dynamicInitializationOnly;
    packet.dynamic_initialization_only =
        dynamicInitializationOnly != 0;
    storage["previous_timestamp"] >> packet.previous_timestamp;
    storage["current_timestamp"] >> packet.current_timestamp;
    storage["search_threshold"] >> packet.search_threshold;
    storage["minimum_matches"] >> packet.minimum_matches;
    storage["feature_count"] >> packet.feature_count;
    storage["previous_feature_count"] >> packet.previous_feature_count;
    storage["scale_levels"] >> packet.scale_levels;
    storage["scale_factor"] >> packet.scale_factor;
    storage["baseline"] >> packet.baseline;
    storage["baseline_fx"] >> packet.baseline_fx;
    storage["fx"] >> packet.fx;
    storage["fy"] >> packet.fy;
    storage["cx"] >> packet.cx;
    storage["cy"] >> packet.cy;
    storage["min_x"] >> packet.min_x;
    storage["max_x"] >> packet.max_x;
    storage["min_y"] >> packet.min_y;
    storage["max_y"] >> packet.max_y;
    storage["grid_width_inverse"] >> packet.grid_width_inverse;
    storage["grid_height_inverse"] >> packet.grid_height_inverse;
    storage["scale_factors"] >> packet.scale_factors;
    storage["inverse_level_sigma2"] >> packet.inverse_level_sigma2;
    if (!readKeypoints(
            storage["current_keypoints"], packet.current_keypoints) ||
        !readKeypoints(
            storage["previous_keypoints"], packet.previous_keypoints)) {
        if (error) *error = "Replay packet has invalid keypoints";
        return false;
    }
    storage["current_right_coordinates"] >>
        packet.current_right_coordinates;
    storage["current_descriptors"] >> packet.current_descriptors;
    packet.previous_map_point_ids.clear();
    for (const cv::FileNode& item :
         storage["previous_map_point_ids"]) {
        std::uint64_t id = 0;
        if (!stringToUint64(item, id)) {
            if (error) *error = "Invalid previous MapPoint ID";
            return false;
        }
        packet.previous_map_point_ids.push_back(id);
    }
    packet.previous_outliers.clear();
    for (const cv::FileNode& item : storage["previous_outliers"]) {
        int outlier = 0;
        item >> outlier;
        packet.previous_outliers.push_back(
            static_cast<unsigned char>(outlier != 0));
    }
    packet.map_points.clear();
    for (const cv::FileNode& item : storage["map_points"]) {
        FrozenReplayMapPoint point;
        std::vector<float> position;
        if (!stringToUint64(item["id"], point.id)) {
            if (error) *error = "Invalid replay MapPoint ID";
            return false;
        }
        item["world_position"] >> position;
        item["observations"] >> point.observations;
        item["descriptor"] >> point.descriptor;
        if (position.size() != 3) {
            if (error) *error = "Invalid replay MapPoint position";
            return false;
        }
        point.world_position =
            Eigen::Vector3f(position[0], position[1], position[2]);
        packet.map_points.push_back(point);
    }
    int localMapSnapshotValid = 0;
    storage["local_map_snapshot_valid"] >> localMapSnapshotValid;
    packet.local_map_snapshot_valid = localMapSnapshotValid != 0;
    packet.local_map_point_ids.clear();
    for (const cv::FileNode& item : storage["local_map_point_ids"]) {
        std::uint64_t id = 0;
        if (!stringToUint64(item, id)) {
            if (error) *error = "Invalid local-map MapPoint ID";
            return false;
        }
        packet.local_map_point_ids.push_back(id);
    }
    packet.local_map_outliers.clear();
    for (const cv::FileNode& item : storage["local_map_outliers"]) {
        int outlier = 0;
        item >> outlier;
        packet.local_map_outliers.push_back(
            static_cast<unsigned char>(outlier != 0));
    }
    if (!stringToUint64(
            storage["local_map_support_hash"],
            packet.local_map_support_hash)) {
        if (error) *error = "Invalid local-map support hash";
        return false;
    }
    storage["previous_gray"] >> packet.previous_gray;
    storage["current_gray"] >> packet.current_gray;
    storage["previous_depth"] >> packet.previous_depth;
    storage["current_depth"] >> packet.current_depth;
    storage["previous_static_mask"] >> packet.previous_static_mask;
    storage["current_static_mask"] >> packet.current_static_mask;
    if (!stringToUint64(
            storage["previous_gray_hash"], packet.previous_gray_hash) ||
        !stringToUint64(
            storage["current_gray_hash"], packet.current_gray_hash) ||
        !stringToUint64(
            storage["previous_depth_hash"], packet.previous_depth_hash) ||
        !stringToUint64(
            storage["current_depth_hash"], packet.current_depth_hash) ||
        !stringToUint64(
            storage["previous_static_mask_hash"],
            packet.previous_static_mask_hash) ||
        !stringToUint64(
            storage["current_static_mask_hash"],
            packet.current_static_mask_hash)) {
        if (error) *error = "Invalid replay asset hash";
        return false;
    }
    const cv::FileNode gate = storage["gate"];
    int useDirect = 0;
    int requireCommon = 0;
    int bypassReliability = 0;
    const cv::FileNode gatePolicy = gate["policy"];
    if (!gatePolicy.empty()) {
        std::string policyName;
        gatePolicy >> policyName;
        if (!parseReplayGatePolicy(policyName, packet.gate.policy)) {
            if (error) *error = "Replay packet has invalid gate policy";
            return false;
        }
    }
    gate["min_inlier_gain"] >> packet.gate.min_inlier_gain;
    gate["min_inlier_gain_ratio"] >>
        packet.gate.min_inlier_gain_ratio;
    gate["max_static_inliers"] >> packet.gate.max_static_inliers;
    gate["min_translation_innovation"] >>
        packet.gate.min_translation_innovation;
    gate["max_translation_innovation"] >>
        packet.gate.max_translation_innovation;
    gate["max_rotation_innovation"] >>
        packet.gate.max_rotation_innovation;
    gate["use_direct_validation"] >> useDirect;
    gate["direct_score_mode"] >> packet.gate.direct_score_mode;
    gate["require_common_support_improvement"] >> requireCommon;
    gate["bypass_reliability_gate"] >> bypassReliability;
    packet.gate.use_direct_validation = useDirect != 0;
    packet.gate.require_common_support_improvement =
        requireCommon != 0;
    packet.gate.bypass_reliability_gate =
        bypassReliability != 0;
    const cv::FileNode direct = storage["direct_config"];
    direct["grid_step"] >> packet.direct_config.grid_step;
    direct["min_common_support"] >>
        packet.direct_config.min_common_support;
    direct["min_depth"] >> packet.direct_config.min_depth;
    direct["max_depth"] >> packet.direct_config.max_depth;
    direct["depth_sigma_constant"] >>
        packet.direct_config.depth_sigma_constant;
    direct["depth_sigma_linear"] >>
        packet.direct_config.depth_sigma_linear;
    direct["photometric_sigma"] >>
        packet.direct_config.photometric_sigma;
    direct["depth_weight"] >> packet.direct_config.depth_weight;
    direct["photometric_weight"] >>
        packet.direct_config.photometric_weight;
    if (!readBranch(
            storage["expected_static"], packet.expected_static) ||
        !readBranch(
            storage["expected_dynamic"], packet.expected_dynamic) ||
        !readDirectResult(
            storage["expected_direct"], packet.expected_direct)) {
        if (error) *error = "Invalid expected replay result";
        return false;
    }
    storage["expected_common_support"] >>
        packet.expected_common_support;
    storage["expected_static_common_score"] >>
        packet.expected_static_common_score;
    storage["expected_dynamic_common_score"] >>
        packet.expected_dynamic_common_score;
    int consensusPass = 0;
    int directPass = 0;
    int commonPass = 0;
    int wouldUse = 0;
    storage["expected_consensus_pass"] >> consensusPass;
    storage["expected_direct_pass"] >> directPass;
    storage["expected_common_support_pass"] >> commonPass;
    storage["expected_would_use"] >> wouldUse;
    packet.expected_consensus_pass = consensusPass != 0;
    packet.expected_direct_pass = directPass != 0;
    packet.expected_common_support_pass = commonPass != 0;
    packet.expected_would_use = wouldUse != 0;
    storage["shadow_translation_blend"] >>
        packet.shadow_translation_blend;
    storage["posterior_min_score_improvement"] >>
        packet.posterior_min_score_improvement;
    if (!readBranch(
            storage["expected_posterior_static"],
            packet.expected_posterior_static) ||
        !readBranch(
            storage["expected_posterior_dynamic"],
            packet.expected_posterior_dynamic)) {
        if (error) *error = "Invalid expected posterior replay result";
        return false;
    }
    storage["expected_posterior_common_support"] >>
        packet.expected_posterior_common_support;
    storage["expected_posterior_static_score"] >>
        packet.expected_posterior_static_score;
    storage["expected_posterior_dynamic_score"] >>
        packet.expected_posterior_dynamic_score;
    int posteriorValid = 0;
    int posteriorPass = 0;
    storage["expected_posterior_valid"] >> posteriorValid;
    storage["expected_posterior_pass"] >> posteriorPass;
    packet.expected_posterior_valid = posteriorValid != 0;
    packet.expected_posterior_pass = posteriorPass != 0;
    int posteriorEvaluated = 0;
    storage["posterior_evaluated"] >> posteriorEvaluated;
    packet.posterior_evaluated = posteriorEvaluated != 0;
    if (!readPose(
            storage["posterior_static_initial_tcw"],
            packet.posterior_static_initial_tcw) ||
        !readPose(
            storage["posterior_dynamic_prior_tcw"],
            packet.posterior_dynamic_prior_tcw) ||
        !readMatrix6(
            storage["posterior_dynamic_information"],
            packet.posterior_dynamic_information)) {
        if (error) *error = "Invalid posterior replay inputs";
        return false;
    }
    std::string validationError;
    if (!validatePacket(packet, validationError)) {
        if (error) *error = validationError;
        return false;
    }
    return true;
}

FrozenReplayResult runFrozenReplay(const FrozenReplayPacket& packet) {
    FrozenReplayResult result;
    if (!validatePacket(packet, result.error)) {
        return result;
    }
    Frame::fx = packet.fx;
    Frame::fy = packet.fy;
    Frame::cx = packet.cx;
    Frame::cy = packet.cy;
    Frame::invfx = 1.0f / packet.fx;
    Frame::invfy = 1.0f / packet.fy;
    Frame::mnMinX = packet.min_x;
    Frame::mnMaxX = packet.max_x;
    Frame::mnMinY = packet.min_y;
    Frame::mnMaxY = packet.max_y;
    Frame::mfGridElementWidthInv = packet.grid_width_inverse;
    Frame::mfGridElementHeightInv = packet.grid_height_inverse;
    ORB_SLAM3::Pinhole camera(
        {packet.fx, packet.fy, packet.cx, packet.cy});

    std::vector<std::unique_ptr<MapPoint>> ownedPoints;
    std::unordered_map<std::uint64_t, MapPoint*> pointsById;
    for (const FrozenReplayMapPoint& state : packet.map_points) {
        std::unique_ptr<MapPoint> point(new MapPoint());
        point->mnId = state.id;
        point->nObs = state.observations;
        point->SetWorldPos(state.world_position);
        point->SetDescriptorForReplay(state.descriptor);
        pointsById[state.id] = point.get();
        ownedPoints.push_back(std::move(point));
    }
    Frame previous;
    previous.N = packet.previous_feature_count;
    previous.Nleft = -1;
    previous.mvKeys = packet.previous_keypoints;
    previous.mvKeysUn = packet.previous_keypoints;
    previous.mvpMapPoints.assign(previous.N, nullptr);
    previous.mvbOutlier.assign(previous.N, false);
    previous.SetPose(packet.previous_tcw);
    for (int index = 0; index < previous.N; ++index) {
        const std::uint64_t id = packet.previous_map_point_ids[index];
        if (id != kNoReplayMapPoint) {
            const auto found = pointsById.find(id);
            if (found == pointsById.end()) {
                result.error = "Previous frame references missing MapPoint";
                return result;
            }
            previous.mvpMapPoints[index] = found->second;
        }
        previous.mvbOutlier[index] =
            packet.previous_outliers[index] != 0;
    }

    const auto runBranch = [&](
        const Sophus::SE3f& initial,
        bool applyDynamicPrior) {
        std::unique_ptr<Frame> frame =
            makeCurrentFrame(packet, &camera, initial);
        ORB_SLAM3::ORBmatcher matcher(0.9f, true);
        int matches = matcher.SearchByProjection(
            *frame, previous, packet.search_threshold, false);
        if (matches < packet.minimum_matches) {
            std::fill(
                frame->mvpMapPoints.begin(),
                frame->mvpMapPoints.end(), nullptr);
            matches = matcher.SearchByProjection(
                *frame, previous, 2 * packet.search_threshold, false);
        }
        FrozenReplayBranchResult branch;
        branch.matches = matches;
        if (applyDynamicPrior &&
            !packet.dynamic_initialization_only) {
            const Eigen::Matrix<float, 6, 6> information =
                packet.dynamic_information_scale *
                packet.dynamic_pose_information;
            branch.inliers = ORB_SLAM3::Optimizer::PoseOptimization(
                frame.get(), &initial, &information);
        } else {
            branch.inliers =
                ORB_SLAM3::Optimizer::PoseOptimization(frame.get());
        }
        branch.optimized_tcw = frame->GetPose();
        return std::make_pair(std::move(frame), branch);
    };

    auto staticRun = runBranch(packet.static_initial_tcw, false);
    auto dynamicRun = runBranch(packet.dynamic_initial_tcw, true);
    result.static_branch = staticRun.second;
    result.dynamic_branch = dynamicRun.second;
    const std::vector<MapPoint*>& staticMatches =
        staticRun.first->mvpMapPoints;
    const std::vector<MapPoint*>& dynamicMatches =
        dynamicRun.first->mvpMapPoints;
    std::vector<ReplaySupportObservation> commonObservations;
    scoreCommonSupport(
        *staticRun.first, staticMatches, dynamicMatches,
        result.static_branch.optimized_tcw,
        result.dynamic_branch.optimized_tcw,
        result.common_support, result.static_common_score,
        result.dynamic_common_score, commonObservations);
    result.static_support_hash = hashSupportObservations(
        supportObservations(*staticRun.first, staticMatches));
    result.dynamic_support_hash = hashSupportObservations(
        supportObservations(*dynamicRun.first, dynamicMatches));
    result.common_support_hash =
        hashSupportObservations(commonObservations);

    result.consensus_pass = evaluateConsensusGate(
        packet.gate, result.static_branch.inliers,
        result.dynamic_branch.inliers,
        result.static_branch.optimized_tcw,
        result.dynamic_branch.optimized_tcw);
    const bool requiresDirect =
        packet.gate.policy == ReplayGatePolicy::DirectCombined ||
        packet.gate.use_direct_validation;
    if (packet.gate.bypass_reliability_gate &&
        packet.gate.policy == ReplayGatePolicy::LegacyConjunction) {
        result.direct_pass = true;
    } else if (requiresDirect) {
        DirectRgbdPoseValidator validator(packet.direct_config);
        DirectRgbdPoseValidator::Intrinsics intrinsics{
            packet.fx, packet.fy, packet.cx, packet.cy};
        result.direct = validator.score(
            packet.previous_gray, packet.current_gray,
            packet.previous_depth, packet.current_depth,
            packet.previous_static_mask, packet.current_static_mask,
            packet.previous_tcw,
            result.static_branch.optimized_tcw,
            result.dynamic_branch.optimized_tcw, intrinsics);
        const DirectRgbdPoseValidator::ScoreMode scoreMode =
            packet.gate.policy == ReplayGatePolicy::DirectCombined
            ? DirectRgbdPoseValidator::ScoreMode::Combined
            : static_cast<DirectRgbdPoseValidator::ScoreMode>(
                  packet.gate.direct_score_mode);
        result.direct_pass =
            DirectRgbdPoseValidator::prefersDynamic(
                result.direct, scoreMode);
    } else {
        result.direct_pass = true;
    }
    result.common_support_pass =
        packet.gate.bypass_reliability_gate ||
        !packet.gate.require_common_support_improvement ||
        (result.common_support > 0 &&
         std::isfinite(result.static_common_score) &&
         std::isfinite(result.dynamic_common_score) &&
         result.static_common_score >= 0.0f &&
         result.dynamic_common_score < result.static_common_score);
    result.would_use = evaluateReplayGatePolicy(
        packet.gate,
        result.static_branch.optimized_tcw,
        result.dynamic_branch.optimized_tcw,
        result.direct,
        result.consensus_pass,
        result.direct_pass,
        result.common_support_pass);
    result.valid = true;
    return result;
}

FrozenReplayPosteriorResult runFrozenReplayPosterior(
    const FrozenReplayPacket& packet,
    const Sophus::SE3f& static_initial_tcw,
    const Sophus::SE3f& dynamic_prior_tcw,
    const Eigen::Matrix<float, 6, 6>& dynamic_information,
    float dynamic_information_scale,
    float minimum_score_improvement,
    float max_static_information_leverage,
    TranslationLeverageMode leverage_mode,
    bool compute_dynamic) {
    FrozenReplayPosteriorResult result;
    if (!validatePacket(packet, result.error)) {
        return result;
    }
    if (!packet.local_map_snapshot_valid) {
        result.error = "Replay packet has no frozen local-map snapshot";
        return result;
    }
    if (!static_initial_tcw.matrix().allFinite() ||
        !dynamic_prior_tcw.matrix().allFinite() ||
        !dynamic_information.allFinite() ||
        !std::isfinite(dynamic_information_scale) ||
        dynamic_information_scale < 0.0f ||
        !std::isfinite(minimum_score_improvement) ||
        minimum_score_improvement < 0.0f ||
        !std::isfinite(max_static_information_leverage) ||
        max_static_information_leverage < 0.0f ||
        (leverage_mode != TranslationLeverageMode::CapOnly &&
         leverage_mode != TranslationLeverageMode::NormalizeToTarget)) {
        result.error = "Posterior replay arguments are invalid";
        return result;
    }

    Frame::fx = packet.fx;
    Frame::fy = packet.fy;
    Frame::cx = packet.cx;
    Frame::cy = packet.cy;
    Frame::invfx = 1.0f / packet.fx;
    Frame::invfy = 1.0f / packet.fy;
    Frame::mnMinX = packet.min_x;
    Frame::mnMaxX = packet.max_x;
    Frame::mnMinY = packet.min_y;
    Frame::mnMaxY = packet.max_y;
    Frame::mfGridElementWidthInv = packet.grid_width_inverse;
    Frame::mfGridElementHeightInv = packet.grid_height_inverse;
    ORB_SLAM3::Pinhole camera(
        {packet.fx, packet.fy, packet.cx, packet.cy});

    std::vector<std::unique_ptr<MapPoint>> ownedPoints;
    std::unordered_map<std::uint64_t, MapPoint*> pointsById;
    for (const FrozenReplayMapPoint& state : packet.map_points) {
        std::unique_ptr<MapPoint> point(new MapPoint());
        point->mnId = state.id;
        point->nObs = state.observations;
        point->SetWorldPos(state.world_position);
        point->SetDescriptorForReplay(state.descriptor);
        pointsById[state.id] = point.get();
        ownedPoints.push_back(std::move(point));
    }

    std::unique_ptr<Frame> base =
        makeCurrentFrame(packet, &camera, static_initial_tcw);
    int supportCount = 0;
    for (int index = 0; index < packet.feature_count; ++index) {
        const std::uint64_t id = packet.local_map_point_ids[index];
        if (id != kNoReplayMapPoint) {
            const auto found = pointsById.find(id);
            if (found == pointsById.end()) {
                result.error =
                    "Posterior replay references missing local MapPoint";
                return result;
            }
            base->mvpMapPoints[index] = found->second;
            ++supportCount;
        }
        base->mvbOutlier[index] =
            packet.local_map_outliers[index] != 0;
    }

    std::unique_ptr<Frame> staticFrame =
        makeCurrentFrame(packet, &camera, static_initial_tcw);
    staticFrame->mvpMapPoints = base->mvpMapPoints;
    staticFrame->mvbOutlier = base->mvbOutlier;
    result.static_branch.matches = supportCount;
    Eigen::Matrix<float, 6, 6> staticInformation =
        Eigen::Matrix<float, 6, 6>::Zero();
    result.static_branch.inliers =
        ORB_SLAM3::Optimizer::PoseOptimization(
            staticFrame.get(), nullptr, nullptr,
            max_static_information_leverage > 0.0f
                ? &staticInformation : nullptr);
    result.static_branch.optimized_tcw = staticFrame->GetPose();

    const auto makeState = [&packet](
        const Frame& frame,
        const std::vector<FrozenReplayMapPoint>& pointStates) {
        FrozenReplayFrameState state;
        state.valid = true;
        state.tcw = frame.GetPose();
        state.keypoints = packet.current_keypoints;
        state.map_point_ids.assign(
            frame.mvpMapPoints.size(), kNoReplayMapPoint);
        state.outliers.assign(frame.mvbOutlier.size(), 0);
        for (std::size_t index = 0;
             index < frame.mvpMapPoints.size(); ++index) {
            if (frame.mvpMapPoints[index])
                state.map_point_ids[index] =
                    frame.mvpMapPoints[index]->mnId;
            state.outliers[index] =
                frame.mvbOutlier[index] ? 1 : 0;
        }
        state.map_points = pointStates;
        return state;
    };
    result.static_frame =
        makeState(*staticFrame, packet.map_points);
    if (!compute_dynamic) {
        result.valid = true;
        return result;
    }

    std::unique_ptr<Frame> dynamicFrame =
        makeCurrentFrame(packet, &camera, dynamic_prior_tcw);
    dynamicFrame->mvpMapPoints = base->mvpMapPoints;
    dynamicFrame->mvbOutlier = base->mvbOutlier;
    const Eigen::Matrix<float, 6, 6> rawInformation =
        dynamic_information_scale * dynamic_information;
    const Eigen::Matrix<float, 6, 6> information =
        max_static_information_leverage > 0.0f
        ? leverageCalibratedTranslationInformation(
              staticInformation, rawInformation,
              max_static_information_leverage, leverage_mode,
              &result.leverage)
        : rawInformation;
    result.dynamic_branch.matches = supportCount;
    result.dynamic_branch.inliers =
        ORB_SLAM3::Optimizer::PoseOptimization(
            dynamicFrame.get(), &dynamic_prior_tcw, &information);
    result.dynamic_branch.optimized_tcw = dynamicFrame->GetPose();

    std::vector<ReplaySupportObservation> commonObservations;
    scoreCommonSupport(
        *base, base->mvpMapPoints, base->mvpMapPoints,
        result.static_branch.optimized_tcw,
        result.dynamic_branch.optimized_tcw,
        result.common_support, result.static_score,
        result.dynamic_score, commonObservations);
    result.gate_valid =
        result.common_support > 0 &&
        std::isfinite(result.static_score) &&
        std::isfinite(result.dynamic_score) &&
        result.static_score >= 0.0f &&
        result.dynamic_score >= 0.0f;
    result.pass =
        result.gate_valid &&
        posteriorLocalMapGate(
            result.static_branch.inliers,
            result.dynamic_branch.inliers,
            result.static_score, result.dynamic_score,
            minimum_score_improvement);

    result.dynamic_frame =
        makeState(*dynamicFrame, packet.map_points);
    result.valid = true;
    return result;
}

FrozenReplayPosteriorResult runFrozenReplayPosterior(
    const FrozenReplayPacket& packet) {
    FrozenReplayPosteriorResult result;
    if (!packet.posterior_evaluated) {
        result.error = "Replay packet has no evaluated posterior";
        return result;
    }
    return runFrozenReplayPosterior(
        packet,
        packet.posterior_static_initial_tcw,
        packet.posterior_dynamic_prior_tcw,
        packet.posterior_dynamic_information,
        1.0f,
        packet.posterior_min_score_improvement,
        0.0f,
        TranslationLeverageMode::CapOnly);
}

FrozenReplayBranchStateResult runFrozenReplayBranch(
    const FrozenReplayPacket& packet,
    const FrozenReplayFrameState& previousState,
    const Sophus::SE3f& initialTcw,
    bool applyDynamicPrior) {
    FrozenReplayBranchStateResult result;
    if (!validatePacket(packet, result.error)) {
        return result;
    }
    if (!initialTcw.matrix().allFinite()) {
        result.error = "Replay branch initial pose is invalid";
        return result;
    }

    const std::vector<cv::KeyPoint>& previousKeypoints =
        previousState.valid
        ? previousState.keypoints
        : packet.previous_keypoints;
    const std::vector<std::uint64_t>& previousPointIds =
        previousState.valid
        ? previousState.map_point_ids
        : packet.previous_map_point_ids;
    const std::vector<unsigned char>& previousOutliers =
        previousState.valid
        ? previousState.outliers
        : packet.previous_outliers;
    const Sophus::SE3f previousTcw =
        previousState.valid ? previousState.tcw : packet.previous_tcw;
    if (previousKeypoints.size() != previousPointIds.size() ||
        previousKeypoints.size() != previousOutliers.size()) {
        result.error = "Replay branch previous-frame arrays are inconsistent";
        return result;
    }
    if (previousState.valid) {
        if (previousKeypoints.size() != packet.previous_keypoints.size()) {
            result.error =
                "Replay branch is not contiguous with the next packet";
            return result;
        }
        for (std::size_t index = 0;
             index < previousKeypoints.size(); ++index) {
            const cv::KeyPoint& carried = previousKeypoints[index];
            const cv::KeyPoint& expected =
                packet.previous_keypoints[index];
            if (carried.pt.x != expected.pt.x ||
                carried.pt.y != expected.pt.y ||
                carried.angle != expected.angle ||
                carried.octave != expected.octave) {
                result.error =
                    "Replay branch keypoints do not match packet continuity";
                return result;
            }
        }
    }

    Frame::fx = packet.fx;
    Frame::fy = packet.fy;
    Frame::cx = packet.cx;
    Frame::cy = packet.cy;
    Frame::invfx = 1.0f / packet.fx;
    Frame::invfy = 1.0f / packet.fy;
    Frame::mnMinX = packet.min_x;
    Frame::mnMaxX = packet.max_x;
    Frame::mnMinY = packet.min_y;
    Frame::mnMaxY = packet.max_y;
    Frame::mfGridElementWidthInv = packet.grid_width_inverse;
    Frame::mfGridElementHeightInv = packet.grid_height_inverse;
    ORB_SLAM3::Pinhole camera(
        {packet.fx, packet.fy, packet.cx, packet.cy});

    std::vector<FrozenReplayMapPoint> pointStates =
        previousState.valid
        ? previousState.map_points
        : packet.map_points;
    std::unordered_map<std::uint64_t, std::size_t> stateIndex;
    for (std::size_t index = 0; index < pointStates.size(); ++index) {
        stateIndex[pointStates[index].id] = index;
    }
    for (const FrozenReplayMapPoint& state : packet.map_points) {
        const auto found = stateIndex.find(state.id);
        if (found == stateIndex.end()) {
            stateIndex[state.id] = pointStates.size();
            pointStates.push_back(state);
            continue;
        }
        FrozenReplayMapPoint& carried = pointStates[found->second];
        if (!carried.world_position.isApprox(
                state.world_position, 0.0f) ||
            carried.observations != state.observations ||
            carried.descriptor.size() != state.descriptor.size() ||
            carried.descriptor.type() != state.descriptor.type() ||
            cv::norm(
                carried.descriptor, state.descriptor,
                cv::NORM_INF) != 0.0) {
            result.error =
                "Replay branch MapPoint state changed across frozen packets";
            return result;
        }
    }
    std::sort(
        pointStates.begin(), pointStates.end(),
        [](const FrozenReplayMapPoint& left,
           const FrozenReplayMapPoint& right) {
            return left.id < right.id;
        });

    std::vector<std::unique_ptr<MapPoint>> ownedPoints;
    std::unordered_map<std::uint64_t, MapPoint*> pointsById;
    for (const FrozenReplayMapPoint& state : pointStates) {
        std::unique_ptr<MapPoint> point(new MapPoint());
        point->mnId = state.id;
        point->nObs = state.observations;
        point->SetWorldPos(state.world_position);
        point->SetDescriptorForReplay(state.descriptor);
        pointsById[state.id] = point.get();
        ownedPoints.push_back(std::move(point));
    }

    Frame previous;
    previous.N = static_cast<int>(previousKeypoints.size());
    previous.Nleft = -1;
    previous.mvKeys = previousKeypoints;
    previous.mvKeysUn = previousKeypoints;
    previous.mvpMapPoints.assign(previous.N, nullptr);
    previous.mvbOutlier.assign(previous.N, false);
    previous.SetPose(previousTcw);
    for (int index = 0; index < previous.N; ++index) {
        const std::uint64_t id = previousPointIds[index];
        if (id != kNoReplayMapPoint) {
            const auto found = pointsById.find(id);
            if (found == pointsById.end()) {
                result.error =
                    "Replay branch references missing carried MapPoint";
                return result;
            }
            previous.mvpMapPoints[index] = found->second;
        }
        previous.mvbOutlier[index] = previousOutliers[index] != 0;
    }

    std::unique_ptr<Frame> current =
        makeCurrentFrame(packet, &camera, initialTcw);
    ORB_SLAM3::ORBmatcher matcher(0.9f, true);
    int matches = matcher.SearchByProjection(
        *current, previous, packet.search_threshold, false);
    if (matches < packet.minimum_matches) {
        std::fill(
            current->mvpMapPoints.begin(),
            current->mvpMapPoints.end(), nullptr);
        matches = matcher.SearchByProjection(
            *current, previous, 2 * packet.search_threshold, false);
    }
    result.branch.matches = matches;
    if (applyDynamicPrior &&
        !packet.dynamic_initialization_only) {
        const Eigen::Matrix<float, 6, 6> information =
            packet.dynamic_information_scale *
            packet.dynamic_pose_information;
        result.branch.inliers =
            ORB_SLAM3::Optimizer::PoseOptimization(
                current.get(), &initialTcw, &information);
    } else {
        result.branch.inliers =
            ORB_SLAM3::Optimizer::PoseOptimization(current.get());
    }
    result.branch.optimized_tcw = current->GetPose();
    result.frame.valid = true;
    result.frame.tcw = result.branch.optimized_tcw;
    result.frame.keypoints = packet.current_keypoints;
    result.frame.map_point_ids.assign(
        current->mvpMapPoints.size(), kNoReplayMapPoint);
    result.frame.outliers.assign(current->mvbOutlier.size(), 0);
    for (std::size_t index = 0;
         index < current->mvpMapPoints.size(); ++index) {
        if (current->mvpMapPoints[index]) {
            result.frame.map_point_ids[index] =
                current->mvpMapPoints[index]->mnId;
        }
        result.frame.outliers[index] =
            current->mvbOutlier[index] ? 1 : 0;
    }
    result.frame.map_points = std::move(pointStates);
    result.valid = true;
    return result;
}

}  // namespace Motion3D
