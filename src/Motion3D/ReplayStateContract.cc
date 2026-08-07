#include "Motion3D/ReplayStateContract.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <tuple>

namespace Motion3D {
namespace {

constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

class Fnv1a64LittleEndian {
public:
    void appendU8(std::uint8_t value) {
        appendByte(value);
    }

    void appendU32(std::uint32_t value) {
        for (int shift = 0; shift < 32; shift += 8) {
            appendByte(static_cast<std::uint8_t>(value >> shift));
        }
    }

    void appendU64(std::uint64_t value) {
        for (int shift = 0; shift < 64; shift += 8) {
            appendByte(static_cast<std::uint8_t>(value >> shift));
        }
    }

    void appendI32(std::int32_t value) {
        appendU32(static_cast<std::uint32_t>(value));
    }

    void appendFloat(float value) {
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "Unexpected float size");
        std::memcpy(&bits, &value, sizeof(bits));
        appendU32(bits);
    }

    std::uint64_t value() const {
        return value_;
    }

private:
    void appendByte(std::uint8_t byte) {
        value_ ^= byte;
        value_ *= kFnvPrime;
    }

    std::uint64_t value_ = kFnvOffsetBasis;
};

void appendPosition(
    Fnv1a64LittleEndian& hash,
    const Eigen::Vector3f& position) {
    hash.appendFloat(position.x());
    hash.appendFloat(position.y());
    hash.appendFloat(position.z());
}

void appendPose(Fnv1a64LittleEndian& hash, const Sophus::SE3f& pose) {
    const Eigen::Quaternionf quaternion = pose.unit_quaternion();
    appendPosition(hash, pose.translation());
    hash.appendFloat(quaternion.x());
    hash.appendFloat(quaternion.y());
    hash.appendFloat(quaternion.z());
    hash.appendFloat(quaternion.w());
}

}  // namespace

Sophus::SE3f canonicalPreviousTcw(
    const Sophus::SE3f& frame_from_reference,
    const Sophus::SE3f& reference_tcw) {
    return frame_from_reference * reference_tcw;
}

std::uint64_t hashPose(const Sophus::SE3f& pose) {
    Fnv1a64LittleEndian hash;
    appendPose(hash, pose);
    return hash.value();
}

std::uint64_t hashCvMat(const cv::Mat& matrix) {
    Fnv1a64LittleEndian hash;
    hash.appendI32(matrix.rows);
    hash.appendI32(matrix.cols);
    hash.appendI32(matrix.type());
    hash.appendI32(matrix.channels());
    if (matrix.empty()) {
        return hash.value();
    }
    const std::size_t rowBytes =
        static_cast<std::size_t>(matrix.cols) * matrix.elemSize();
    hash.appendU64(rowBytes);
    for (int row = 0; row < matrix.rows; ++row) {
        const std::uint8_t* data = matrix.ptr<std::uint8_t>(row);
        for (std::size_t index = 0; index < rowBytes; ++index) {
            hash.appendU8(data[index]);
        }
    }
    return hash.value();
}

std::uint64_t hashMapPointStates(
    const std::vector<ReplayMapPointState>& states) {
    std::vector<ReplayMapPointState> sorted = states;
    std::sort(
        sorted.begin(), sorted.end(),
        [](const ReplayMapPointState& left,
           const ReplayMapPointState& right) {
            return left.map_point_id < right.map_point_id;
        });
    Fnv1a64LittleEndian hash;
    hash.appendU64(sorted.size());
    for (const ReplayMapPointState& state : sorted) {
        hash.appendU64(state.map_point_id);
        appendPosition(hash, state.world_position);
    }
    return hash.value();
}

std::uint64_t hashMapPointTrackingStates(
    const std::vector<ReplayMapPointTrackingState>& states) {
    return digestMapPointTrackingStates(states).state_hash;
}

ReplayMapPointTrackingDigest digestMapPointTrackingStates(
    const std::vector<ReplayMapPointTrackingState>& states) {
    std::vector<ReplayMapPointTrackingState> sorted = states;
    std::sort(
        sorted.begin(), sorted.end(),
        [](const ReplayMapPointTrackingState& left,
           const ReplayMapPointTrackingState& right) {
            return left.map_point_id < right.map_point_id;
        });
    Fnv1a64LittleEndian stateHash;
    Fnv1a64LittleEndian idHash;
    Fnv1a64LittleEndian positionHash;
    Fnv1a64LittleEndian normalHash;
    Fnv1a64LittleEndian distanceHash;
    Fnv1a64LittleEndian descriptorHash;
    Fnv1a64LittleEndian observationHash;
    stateHash.appendU64(sorted.size());
    idHash.appendU64(sorted.size());
    positionHash.appendU64(sorted.size());
    normalHash.appendU64(sorted.size());
    distanceHash.appendU64(sorted.size());
    descriptorHash.appendU64(sorted.size());
    observationHash.appendU64(sorted.size());
    for (const ReplayMapPointTrackingState& state : sorted) {
        stateHash.appendU64(state.map_point_id);
        appendPosition(stateHash, state.world_position);
        appendPosition(stateHash, state.normal);
        stateHash.appendFloat(state.min_distance);
        stateHash.appendFloat(state.max_distance);
        stateHash.appendU64(state.descriptor_hash);
        stateHash.appendI32(state.observations);

        idHash.appendU64(state.map_point_id);

        positionHash.appendU64(state.map_point_id);
        appendPosition(positionHash, state.world_position);

        normalHash.appendU64(state.map_point_id);
        appendPosition(normalHash, state.normal);

        distanceHash.appendU64(state.map_point_id);
        distanceHash.appendFloat(state.min_distance);
        distanceHash.appendFloat(state.max_distance);

        descriptorHash.appendU64(state.map_point_id);
        descriptorHash.appendU64(state.descriptor_hash);

        observationHash.appendU64(state.map_point_id);
        observationHash.appendI32(state.observations);
    }
    ReplayMapPointTrackingDigest digest;
    digest.state_hash = stateHash.value();
    digest.id_hash = idHash.value();
    digest.position_hash = positionHash.value();
    digest.normal_hash = normalHash.value();
    digest.distance_hash = distanceHash.value();
    digest.descriptor_hash = descriptorHash.value();
    digest.observation_hash = observationHash.value();
    return digest;
}

std::uint64_t hashKeyFrameStates(
    const std::vector<ReplayKeyFrameState>& states) {
    std::vector<ReplayKeyFrameState> sorted = states;
    std::sort(
        sorted.begin(), sorted.end(),
        [](const ReplayKeyFrameState& left,
           const ReplayKeyFrameState& right) {
            return left.keyframe_id < right.keyframe_id;
        });
    Fnv1a64LittleEndian hash;
    hash.appendU64(sorted.size());
    for (const ReplayKeyFrameState& state : sorted) {
        hash.appendU64(state.keyframe_id);
        appendPose(hash, state.tcw);
    }
    return hash.value();
}

std::uint64_t hashIdSequence(
    const std::vector<std::uint64_t>& ids) {
    Fnv1a64LittleEndian hash;
    hash.appendU64(ids.size());
    for (const std::uint64_t id : ids) {
        hash.appendU64(id);
    }
    return hash.value();
}

std::uint64_t hashExecutionMapState(
    std::uint64_t map_id,
    std::int64_t map_generation,
    std::uint64_t map_point_hash,
    std::uint64_t keyframe_hash) {
    Fnv1a64LittleEndian hash;
    hash.appendU64(map_id);
    hash.appendU64(static_cast<std::uint64_t>(map_generation));
    hash.appendU64(map_point_hash);
    hash.appendU64(keyframe_hash);
    return hash.value();
}

std::uint64_t hashSupportObservations(
    const std::vector<ReplaySupportObservation>& observations) {
    std::vector<ReplaySupportObservation> sorted = observations;
    std::sort(
        sorted.begin(), sorted.end(),
        [](const ReplaySupportObservation& left,
           const ReplaySupportObservation& right) {
            return std::tie(left.feature_index, left.map_point_id) <
                std::tie(right.feature_index, right.map_point_id);
        });
    Fnv1a64LittleEndian hash;
    hash.appendU64(sorted.size());
    for (const ReplaySupportObservation& observation : sorted) {
        hash.appendU64(observation.feature_index);
        hash.appendU64(observation.map_point_id);
        appendPosition(hash, observation.world_position);
        hash.appendFloat(observation.keypoint_x);
        hash.appendFloat(observation.keypoint_y);
        hash.appendFloat(observation.right_coordinate);
        hash.appendI32(observation.octave);
    }
    return hash.value();
}

std::string hashToHex(std::uint64_t hash) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << hash;
    return stream.str();
}

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
    const std::vector<ReplaySupportObservation>& common_support) {
    ReplayStateIdentity identity;
    identity.valid = previous_tcw.matrix().allFinite() &&
        reference_kf_tcw.matrix().allFinite();
    identity.previous_frame_id = previous_frame_id;
    identity.map_id = map_id;
    identity.map_generation = map_generation;
    identity.reference_kf_id = reference_kf_id;
    identity.previous_tcw = previous_tcw;
    identity.reference_kf_pose_hash = hashPose(reference_kf_tcw);
    identity.map_point_state_hash = hashMapPointStates(map_points);
    identity.static_support_hash = hashSupportObservations(static_support);
    identity.dynamic_support_hash = hashSupportObservations(dynamic_support);
    identity.common_support_hash = hashSupportObservations(common_support);
    return identity;
}

}  // namespace Motion3D
