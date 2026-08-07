#include "Motion3D/FrozenReplayPacket.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

float poseTranslationError(
    const Sophus::SE3f& expected,
    const Sophus::SE3f& actual) {
    return (expected.inverse() * actual).translation().norm();
}

float poseRotationError(
    const Sophus::SE3f& expected,
    const Sophus::SE3f& actual) {
    return (expected.so3().inverse() * actual.so3()).log().norm();
}

std::string poseComponents(const Sophus::SE3f& pose) {
    const Eigen::Quaternionf rotation = pose.unit_quaternion();
    std::ostringstream stream;
    stream << std::setprecision(9)
           << '[' << pose.translation().x()
           << ',' << pose.translation().y()
           << ',' << pose.translation().z()
           << ',' << rotation.x()
           << ',' << rotation.y()
           << ',' << rotation.z()
           << ',' << rotation.w() << ']';
    return stream.str();
}

void reportDifference(
    const std::string& packet,
    const std::string& field,
    const std::string& expected,
    const std::string& actual,
    int& differences) {
    std::cerr << packet << ": " << field
              << " expected=" << expected
              << " actual=" << actual << '\n';
    ++differences;
}

template <typename Value>
void compareExact(
    const std::string& packet,
    const std::string& field,
    const Value& expected,
    const Value& actual,
    int& differences) {
    if (expected != actual) {
        reportDifference(
            packet, field, std::to_string(expected),
            std::to_string(actual), differences);
    }
}

int verifyPacket(const std::string& path) {
    Motion3D::FrozenReplayPacket packet;
    std::string error;
    if (!Motion3D::readFrozenReplayPacket(path, packet, &error)) {
        std::cerr << path << ": " << error << '\n';
        return 1;
    }
    const Motion3D::FrozenReplayResult replay =
        Motion3D::runFrozenReplay(packet);
    if (!replay.valid) {
        std::cerr << path << ": " << replay.error << '\n';
        return 1;
    }

    int differences = 0;
    compareExact(
        path, "static.matches", packet.expected_static.matches,
        replay.static_branch.matches, differences);
    compareExact(
        path, "static.inliers", packet.expected_static.inliers,
        replay.static_branch.inliers, differences);
    compareExact(
        path, "dynamic.matches", packet.expected_dynamic.matches,
        replay.dynamic_branch.matches, differences);
    compareExact(
        path, "dynamic.inliers", packet.expected_dynamic.inliers,
        replay.dynamic_branch.inliers, differences);
    compareExact(
        path, "common_support", packet.expected_common_support,
        replay.common_support, differences);
    compareExact(
        path, "consensus_pass", packet.expected_consensus_pass,
        replay.consensus_pass, differences);
    compareExact(
        path, "direct_pass", packet.expected_direct_pass,
        replay.direct_pass, differences);
    compareExact(
        path, "common_support_pass",
        packet.expected_common_support_pass,
        replay.common_support_pass, differences);
    compareExact(
        path, "would_use", packet.expected_would_use,
        replay.would_use, differences);
    compareExact(
        path, "static_support_hash",
        packet.state_identity.static_support_hash,
        replay.static_support_hash, differences);
    compareExact(
        path, "dynamic_support_hash",
        packet.state_identity.dynamic_support_hash,
        replay.dynamic_support_hash, differences);
    compareExact(
        path, "common_support_hash",
        packet.state_identity.common_support_hash,
        replay.common_support_hash, differences);

    const auto comparePose = [&](const std::string& name,
                                 const Sophus::SE3f& expected,
                                 const Sophus::SE3f& actual) {
        const float translation =
            poseTranslationError(expected, actual);
        const float rotation =
            poseRotationError(expected, actual);
        if (translation >
            Motion3D::kReplayPoseTranslationTolerance) {
            reportDifference(
                path, name + ".translation_error_m", "<=0.00005",
                std::to_string(translation) +
                    " expected_pose=" + poseComponents(expected) +
                    " actual_pose=" + poseComponents(actual),
                differences);
        }
        if (rotation >
            Motion3D::kReplayPoseRotationTolerance) {
            reportDifference(
                path, name + ".rotation_error_rad", "<=0.00001",
                std::to_string(rotation) +
                    " expected_pose=" + poseComponents(expected) +
                    " actual_pose=" + poseComponents(actual),
                differences);
        }
    };
    comparePose(
        "static.optimized_tcw",
        packet.expected_static.optimized_tcw,
        replay.static_branch.optimized_tcw);
    comparePose(
        "dynamic.optimized_tcw",
        packet.expected_dynamic.optimized_tcw,
        replay.dynamic_branch.optimized_tcw);
    if (!Motion3D::replayScoreEquivalent(
            packet.expected_static_common_score,
            replay.static_common_score)) {
        reportDifference(
            path, "static_common_score",
            std::to_string(packet.expected_static_common_score),
            std::to_string(replay.static_common_score), differences);
    }
    if (!Motion3D::replayScoreEquivalent(
            packet.expected_dynamic_common_score,
            replay.dynamic_common_score)) {
        reportDifference(
            path, "dynamic_common_score",
            std::to_string(packet.expected_dynamic_common_score),
            std::to_string(replay.dynamic_common_score), differences);
    }
    if (packet.gate.use_direct_validation) {
        compareExact(
            path, "direct.valid", packet.expected_direct.valid,
            replay.direct.valid, differences);
        compareExact(
            path, "direct.common_support",
            packet.expected_direct.common_support,
            replay.direct.common_support, differences);
        if (!Motion3D::replayDirectScoreEquivalent(
                packet.expected_direct.static_combined_score,
                replay.direct.static_combined_score)) {
            reportDifference(
                path, "direct.static_combined_score",
                std::to_string(
                    packet.expected_direct.static_combined_score),
                std::to_string(replay.direct.static_combined_score),
                differences);
        }
        if (!Motion3D::replayDirectScoreEquivalent(
                packet.expected_direct.dynamic_combined_score,
                replay.direct.dynamic_combined_score)) {
            reportDifference(
                path, "direct.dynamic_combined_score",
                std::to_string(
                    packet.expected_direct.dynamic_combined_score),
                std::to_string(replay.direct.dynamic_combined_score),
                differences);
        }
    }
    if (packet.posterior_evaluated) {
        const Motion3D::FrozenReplayPosteriorResult posterior =
            Motion3D::runFrozenReplayPosterior(packet);
        if (!posterior.valid) {
            reportDifference(
                path, "posterior.execution", "valid",
                posterior.error, differences);
        } else {
            compareExact(
                path, "posterior.static.matches",
                packet.expected_posterior_static.matches,
                posterior.static_branch.matches, differences);
            compareExact(
                path, "posterior.static.inliers",
                packet.expected_posterior_static.inliers,
                posterior.static_branch.inliers, differences);
            compareExact(
                path, "posterior.dynamic.matches",
                packet.expected_posterior_dynamic.matches,
                posterior.dynamic_branch.matches, differences);
            compareExact(
                path, "posterior.dynamic.inliers",
                packet.expected_posterior_dynamic.inliers,
                posterior.dynamic_branch.inliers, differences);
            compareExact(
                path, "posterior.common_support",
                packet.expected_posterior_common_support,
                posterior.common_support, differences);
            compareExact(
                path, "posterior.valid",
                packet.expected_posterior_valid,
                posterior.gate_valid, differences);
            compareExact(
                path, "posterior.pass",
                packet.expected_posterior_pass,
                posterior.pass, differences);
            comparePose(
                "posterior.static.optimized_tcw",
                packet.expected_posterior_static.optimized_tcw,
                posterior.static_branch.optimized_tcw);
            comparePose(
                "posterior.dynamic.optimized_tcw",
                packet.expected_posterior_dynamic.optimized_tcw,
                posterior.dynamic_branch.optimized_tcw);
            if (!Motion3D::replayScoreEquivalent(
                    packet.expected_posterior_static_score,
                    posterior.static_score)) {
                reportDifference(
                    path, "posterior.static_score",
                    std::to_string(
                        packet.expected_posterior_static_score),
                    std::to_string(posterior.static_score),
                    differences);
            }
            if (!Motion3D::replayScoreEquivalent(
                    packet.expected_posterior_dynamic_score,
                    posterior.dynamic_score)) {
                reportDifference(
                    path, "posterior.dynamic_score",
                    std::to_string(
                        packet.expected_posterior_dynamic_score),
                    std::to_string(posterior.dynamic_score),
                    differences);
            }
        }
    }
    if (differences == 0) {
        std::cout << path << ": PASS"
                  << " static=" << replay.static_branch.inliers
                  << " dynamic=" << replay.dynamic_branch.inliers
                  << " would_use=" << replay.would_use
                  << " posterior="
                  << (packet.posterior_evaluated ? "verified" : "none")
                  << '\n';
    }
    return differences == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " packet.yml.gz [packet.yml.gz ...]\n";
        return 2;
    }
    int failures = 0;
    for (int index = 1; index < argc; ++index) {
        failures += verifyPacket(argv[index]);
    }
    if (failures != 0) {
        std::cerr << failures << " replay packet(s) failed\n";
        return 1;
    }
    std::cout << (argc - 1) << " replay packet(s) passed\n";
    return 0;
}
