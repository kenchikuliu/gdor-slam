#include <gtest/gtest.h>

#include "Motion3D/FrozenReplayPacket.h"

#include <filesystem>
#include <limits>
#include <unistd.h>

namespace {

Motion3D::FrozenReplayPacket syntheticPacket() {
    Motion3D::FrozenReplayPacket packet;
    packet.valid = true;
    packet.freeze.requested = true;
    packet.freeze.local_mapping_idle_ack = true;
    packet.freeze.local_mapping_stopped_ack = true;
    packet.freeze.loop_closing_idle_ack = true;
    packet.freeze.loop_closing_stopped_ack = true;
    packet.freeze.gba_stopped_ack = true;
    packet.freeze.epoch = 1;
    packet.state_identity.valid = true;
    packet.state_identity.previous_frame_id = 10;
    packet.state_identity.map_id = 2;
    packet.state_identity.map_generation = 4;
    packet.state_identity.reference_kf_id = 3;
    packet.fx = 80.0f;
    packet.fy = 80.0f;
    packet.cx = 31.5f;
    packet.cy = 23.5f;
    packet.min_x = 0.0f;
    packet.max_x = 64.0f;
    packet.min_y = 0.0f;
    packet.max_y = 48.0f;
    packet.grid_width_inverse = 1.0f;
    packet.grid_height_inverse = 1.0f;
    packet.baseline = 0.075f;
    packet.baseline_fx = 6.0f;
    packet.scale_levels = 1;
    packet.scale_factor = 1.0f;
    packet.scale_factors = {1.0f};
    packet.inverse_level_sigma2 = {1.0f};
    packet.feature_count = 30;
    packet.previous_feature_count = 30;
    packet.current_descriptors =
        cv::Mat(packet.feature_count, 32, CV_8UC1);
    packet.current_right_coordinates.assign(
        packet.feature_count, -1.0f);
    packet.previous_map_point_ids.resize(packet.feature_count);
    packet.previous_outliers.assign(packet.feature_count, 0);
    packet.local_map_snapshot_valid = true;
    packet.local_map_outliers.assign(packet.feature_count, 0);
    std::vector<Motion3D::ReplayMapPointState> mapPointStates;
    std::vector<Motion3D::ReplaySupportObservation> localSupport;
    for (int index = 0; index < packet.feature_count; ++index) {
        const float x = 8.0f + 9.0f * (index % 6);
        const float y = 6.0f + 8.0f * (index / 6);
        cv::KeyPoint keypoint(x, y, 8.0f, 0.0f, 1.0f, 0);
        packet.current_keypoints.push_back(keypoint);
        packet.previous_keypoints.push_back(keypoint);
        packet.current_descriptors.row(index).setTo(
            cv::Scalar(index * 7 + 3));
        Motion3D::FrozenReplayMapPoint point;
        point.id = 100 + index;
        point.world_position = Eigen::Vector3f(
            (x - packet.cx) * 2.0f / packet.fx,
            (y - packet.cy) * 2.0f / packet.fy,
            2.0f);
        point.observations = 1;
        point.descriptor =
            packet.current_descriptors.row(index).clone();
        packet.map_points.push_back(point);
        packet.previous_map_point_ids[index] = point.id;
        packet.local_map_point_ids.push_back(point.id);
        Motion3D::ReplaySupportObservation observation;
        observation.feature_index = index;
        observation.map_point_id = point.id;
        observation.world_position = point.world_position;
        observation.keypoint_x = keypoint.pt.x;
        observation.keypoint_y = keypoint.pt.y;
        observation.right_coordinate = -1.0f;
        observation.octave = keypoint.octave;
        localSupport.push_back(observation);
        mapPointStates.push_back(
            Motion3D::ReplayMapPointState{
                point.id, point.world_position});
    }
    packet.previous_gray = cv::Mat(48, 64, CV_8UC1);
    for (int y = 0; y < packet.previous_gray.rows; ++y) {
        for (int x = 0; x < packet.previous_gray.cols; ++x) {
            packet.previous_gray.at<unsigned char>(y, x) =
                static_cast<unsigned char>(2 * x + y);
        }
    }
    packet.current_gray = packet.previous_gray.clone();
    packet.previous_depth =
        cv::Mat(48, 64, CV_32FC1, cv::Scalar(2.0f));
    packet.current_depth = packet.previous_depth.clone();
    packet.previous_static_mask =
        cv::Mat(48, 64, CV_8UC1, cv::Scalar(255));
    packet.current_static_mask =
        packet.previous_static_mask.clone();
    packet.previous_gray_hash =
        Motion3D::hashCvMat(packet.previous_gray);
    packet.current_gray_hash =
        Motion3D::hashCvMat(packet.current_gray);
    packet.previous_depth_hash =
        Motion3D::hashCvMat(packet.previous_depth);
    packet.current_depth_hash =
        Motion3D::hashCvMat(packet.current_depth);
    packet.previous_static_mask_hash =
        Motion3D::hashCvMat(packet.previous_static_mask);
    packet.current_static_mask_hash =
        Motion3D::hashCvMat(packet.current_static_mask);
    packet.state_identity.map_point_state_hash =
        Motion3D::hashMapPointStates(mapPointStates);
    packet.local_map_support_hash =
        Motion3D::hashSupportObservations(localSupport);
    packet.gate.min_inlier_gain = 0;
    packet.gate.min_inlier_gain_ratio = 0.0f;
    packet.gate.require_common_support_improvement = true;
    packet.gate.use_direct_validation = false;
    packet.dynamic_initialization_only = true;
    packet.dynamic_initial_tcw.translation().x() = 0.01f;
    return packet;
}

}  // namespace

TEST(FrozenReplayPacketTest, ReplaysProjectionOptimizationAndGate) {
    Motion3D::FrozenReplayPacket packet = syntheticPacket();
    const Motion3D::FrozenReplayResult result =
        Motion3D::runFrozenReplay(packet);
    ASSERT_TRUE(result.valid) << result.error;
    EXPECT_EQ(result.static_branch.matches, 30);
    EXPECT_EQ(result.dynamic_branch.matches, 30);
    EXPECT_EQ(result.static_branch.inliers, 30);
    EXPECT_EQ(result.dynamic_branch.inliers, 30);
    EXPECT_EQ(result.common_support, 30);
    EXPECT_EQ(result.static_support_hash, result.dynamic_support_hash);
    EXPECT_TRUE(result.consensus_pass);
    EXPECT_FALSE(result.common_support_pass);
    EXPECT_FALSE(result.would_use);
    EXPECT_LT(
        result.static_branch.optimized_tcw.translation().norm(), 1e-5f);
    EXPECT_LT(
        result.dynamic_branch.optimized_tcw.translation().norm(), 1e-5f);
}

TEST(FrozenReplayPacketTest, UsesBoundedAbsoluteAndRelativeScoreTolerance) {
    EXPECT_TRUE(Motion3D::replayScoreEquivalent(-1.0f, -2.0f));
    EXPECT_TRUE(Motion3D::replayScoreEquivalent(4.981379f, 4.981411f));
    EXPECT_FALSE(Motion3D::replayScoreEquivalent(4.981379f, 4.982379f));
    EXPECT_FALSE(Motion3D::replayScoreEquivalent(
        std::numeric_limits<float>::infinity(), 1.0f));
    EXPECT_TRUE(Motion3D::replayDirectScoreEquivalent(
        0.216690f, 0.216649f));
    EXPECT_FALSE(Motion3D::replayDirectScoreEquivalent(
        0.216690f, 0.217690f));
}

TEST(FrozenReplayPacketTest, ParsesExplicitGatePolicies) {
    Motion3D::ReplayGatePolicy policy =
        Motion3D::ReplayGatePolicy::LegacyConjunction;
    EXPECT_STREQ(
        Motion3D::replayGatePolicyName(policy),
        "legacy-conjunction");
    EXPECT_TRUE(Motion3D::parseReplayGatePolicy(
        "direct-combined", policy));
    EXPECT_EQ(
        static_cast<int>(policy),
        static_cast<int>(
            Motion3D::ReplayGatePolicy::DirectCombined));
    EXPECT_STREQ(
        Motion3D::replayGatePolicyName(policy),
        "direct-combined");
    EXPECT_FALSE(Motion3D::parseReplayGatePolicy(
        "unsupported", policy));
}

TEST(FrozenReplayPacketTest, DirectCombinedGateIsFailClosedAndBounded) {
    Motion3D::ReplayGateConfig config;
    config.policy = Motion3D::ReplayGatePolicy::DirectCombined;
    config.min_translation_innovation = 0.001f;
    config.max_translation_innovation = 0.02f;
    config.max_rotation_innovation = 0.01f;
    const Sophus::SE3f static_tcw;
    Sophus::SE3f dynamic_tcw;
    dynamic_tcw.translation().x() = 0.005f;

    Motion3D::DirectRgbdPoseValidator::Result direct;
    EXPECT_FALSE(Motion3D::evaluateReplayGatePolicy(
        config, static_tcw, dynamic_tcw, direct,
        false, false, false));

    direct.valid = true;
    direct.common_support = 100;
    direct.static_combined_score = 1.0f;
    direct.dynamic_combined_score = 0.9f;
    EXPECT_TRUE(Motion3D::evaluateReplayGatePolicy(
        config, static_tcw, dynamic_tcw, direct,
        false, true, false));

    config.bypass_reliability_gate = true;
    EXPECT_FALSE(Motion3D::evaluateReplayGatePolicy(
        config, static_tcw, dynamic_tcw, direct,
        true, true, true));
    config.bypass_reliability_gate = false;

    dynamic_tcw.translation().x() = 0.03f;
    EXPECT_FALSE(Motion3D::evaluateReplayGatePolicy(
        config, static_tcw, dynamic_tcw, direct,
        true, true, true));
}

TEST(FrozenReplayPacketTest, LegacyGatePolicyPreservesConjunction) {
    Motion3D::ReplayGateConfig config;
    Motion3D::DirectRgbdPoseValidator::Result direct;
    EXPECT_TRUE(Motion3D::evaluateReplayGatePolicy(
        config, Sophus::SE3f(), Sophus::SE3f(), direct,
        true, true, true));
    EXPECT_FALSE(Motion3D::evaluateReplayGatePolicy(
        config, Sophus::SE3f(), Sophus::SE3f(), direct,
        true, false, true));
}

TEST(FrozenReplayPacketTest, ExportsOnlyAfterOneSettlingFrame) {
    EXPECT_FALSE(Motion3D::replayPacketPastFreezeSettlingFrame(200, 200));
    EXPECT_FALSE(Motion3D::replayPacketPastFreezeSettlingFrame(201, 200));
    EXPECT_TRUE(Motion3D::replayPacketPastFreezeSettlingFrame(202, 200));
    EXPECT_FALSE(Motion3D::replayPacketPastFreezeSettlingFrame(202, -1));
}

TEST(FrozenReplayPacketTest, RoundTripsWithoutChangingReplay) {
    Motion3D::FrozenReplayPacket packet = syntheticPacket();
    packet.dynamic_initialization_only = false;
    packet.dynamic_information_scale = 0.25f;
    packet.dynamic_pose_information =
        123.0f * Eigen::Matrix<float, 6, 6>::Identity();
    packet.gate.bypass_reliability_gate = true;
    const Motion3D::FrozenReplayResult first =
        Motion3D::runFrozenReplay(packet);
    ASSERT_TRUE(first.valid) << first.error;
    packet.expected_static = first.static_branch;
    packet.expected_dynamic = first.dynamic_branch;
    packet.expected_common_support = first.common_support;
    packet.expected_static_common_score = first.static_common_score;
    packet.expected_dynamic_common_score = first.dynamic_common_score;
    packet.expected_direct = first.direct;
    packet.expected_consensus_pass = first.consensus_pass;
    packet.expected_direct_pass = first.direct_pass;
    packet.expected_common_support_pass = first.common_support_pass;
    packet.expected_would_use = first.would_use;
    packet.state_identity.static_support_hash =
        first.static_support_hash;
    packet.state_identity.dynamic_support_hash =
        first.dynamic_support_hash;
    packet.state_identity.common_support_hash =
        first.common_support_hash;
    packet.posterior_min_score_improvement = 0.0015f;
    packet.posterior_evaluated = true;
    packet.posterior_static_initial_tcw =
        first.static_branch.optimized_tcw;
    packet.posterior_dynamic_prior_tcw =
        Motion3D::reuseShadowCameraTranslation(
            first.static_branch.optimized_tcw,
            first.dynamic_branch.optimized_tcw, 0.5f);
    packet.posterior_dynamic_information =
        packet.dynamic_information_scale *
        Motion3D::isotropicTranslationInformation(
            packet.dynamic_pose_information);
    const Motion3D::FrozenReplayPosteriorResult firstPosterior =
        Motion3D::runFrozenReplayPosterior(packet);
    ASSERT_TRUE(firstPosterior.valid) << firstPosterior.error;
    packet.expected_posterior_static =
        firstPosterior.static_branch;
    packet.expected_posterior_dynamic =
        firstPosterior.dynamic_branch;
    packet.expected_posterior_common_support =
        firstPosterior.common_support;
    packet.expected_posterior_static_score =
        firstPosterior.static_score;
    packet.expected_posterior_dynamic_score =
        firstPosterior.dynamic_score;
    packet.expected_posterior_valid =
        firstPosterior.gate_valid;
    packet.expected_posterior_pass = firstPosterior.pass;
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("frozen_replay_" + std::to_string(getpid()) + ".yml.gz");
    std::string error;
    ASSERT_TRUE(Motion3D::writeFrozenReplayPacket(
        packet, path.string(), &error)) << error;
    Motion3D::FrozenReplayPacket restored;
    ASSERT_TRUE(Motion3D::readFrozenReplayPacket(
        path.string(), restored, &error)) << error;
    EXPECT_TRUE(restored.freeze.complete());
    EXPECT_EQ(restored.freeze.epoch, 1u);
    EXPECT_EQ(
        restored.freeze.protocol,
        Motion3D::kReplayFreezeProtocolVersion);
    EXPECT_TRUE(restored.gate.bypass_reliability_gate);
    EXPECT_EQ(
        static_cast<int>(restored.gate.policy),
        static_cast<int>(
            Motion3D::ReplayGatePolicy::LegacyConjunction));
    EXPECT_FALSE(restored.dynamic_initialization_only);
    EXPECT_FLOAT_EQ(restored.dynamic_information_scale, 0.25f);
    EXPECT_TRUE(restored.dynamic_pose_information.isApprox(
        packet.dynamic_pose_information, 0.0f));
    EXPECT_TRUE(restored.posterior_evaluated);
    EXPECT_FLOAT_EQ(
        restored.posterior_min_score_improvement, 0.0015f);
    EXPECT_TRUE(
        restored.posterior_static_initial_tcw.matrix().isApprox(
            packet.posterior_static_initial_tcw.matrix(), 0.0f));
    EXPECT_TRUE(
        restored.posterior_dynamic_prior_tcw.matrix().isApprox(
            packet.posterior_dynamic_prior_tcw.matrix(), 0.0f));
    EXPECT_TRUE(restored.posterior_dynamic_information.isApprox(
        packet.posterior_dynamic_information, 0.0f));
    EXPECT_EQ(
        restored.local_map_point_ids,
        packet.local_map_point_ids);
    EXPECT_EQ(
        restored.local_map_outliers,
        packet.local_map_outliers);
    EXPECT_EQ(
        restored.local_map_support_hash,
        packet.local_map_support_hash);
    const Motion3D::FrozenReplayResult second =
        Motion3D::runFrozenReplay(restored);
    std::filesystem::remove(path);
    ASSERT_TRUE(second.valid) << second.error;
    EXPECT_EQ(first.static_branch.matches, second.static_branch.matches);
    EXPECT_EQ(first.dynamic_branch.matches, second.dynamic_branch.matches);
    EXPECT_EQ(first.static_branch.inliers, second.static_branch.inliers);
    EXPECT_EQ(first.dynamic_branch.inliers, second.dynamic_branch.inliers);
    EXPECT_EQ(first.static_support_hash, second.static_support_hash);
    EXPECT_EQ(first.dynamic_support_hash, second.dynamic_support_hash);
    EXPECT_EQ(first.common_support_hash, second.common_support_hash);
    EXPECT_NEAR(
        first.static_common_score, second.static_common_score, 1e-6f);
    EXPECT_NEAR(
        first.dynamic_common_score, second.dynamic_common_score, 1e-6f);
    const Motion3D::FrozenReplayPosteriorResult secondPosterior =
        Motion3D::runFrozenReplayPosterior(restored);
    ASSERT_TRUE(secondPosterior.valid) << secondPosterior.error;
    EXPECT_EQ(
        firstPosterior.static_branch.inliers,
        secondPosterior.static_branch.inliers);
    EXPECT_EQ(
        firstPosterior.dynamic_branch.inliers,
        secondPosterior.dynamic_branch.inliers);
    EXPECT_EQ(
        firstPosterior.common_support,
        secondPosterior.common_support);
    EXPECT_EQ(firstPosterior.gate_valid, secondPosterior.gate_valid);
    EXPECT_EQ(firstPosterior.pass, secondPosterior.pass);
    EXPECT_NEAR(
        firstPosterior.static_score,
        secondPosterior.static_score, 1e-6f);
    EXPECT_NEAR(
        firstPosterior.dynamic_score,
        secondPosterior.dynamic_score, 1e-6f);
}

TEST(FrozenReplayPacketTest, PreservesPoseBitsAcrossSerialization) {
    Motion3D::FrozenReplayPacket packet = syntheticPacket();
    const Sophus::SE3f first(
        Sophus::SO3f::exp(Eigen::Vector3f(0.17f, -0.31f, 0.08f)),
        Eigen::Vector3f(0.12f, -0.04f, 0.63f));
    const Sophus::SE3f second(
        Sophus::SO3f::exp(Eigen::Vector3f(-0.05f, 0.11f, 0.23f)),
        Eigen::Vector3f(-0.09f, 0.21f, 0.07f));
    packet.static_initial_tcw = first * second;
    const std::uint64_t expectedHash =
        Motion3D::hashPose(packet.static_initial_tcw);
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("frozen_replay_pose_bits_" +
         std::to_string(getpid()) + ".yml.gz");

    std::string error;
    ASSERT_TRUE(Motion3D::writeFrozenReplayPacket(
        packet, path.string(), &error)) << error;
    Motion3D::FrozenReplayPacket restored;
    ASSERT_TRUE(Motion3D::readFrozenReplayPacket(
        path.string(), restored, &error)) << error;
    std::filesystem::remove(path);

    EXPECT_EQ(
        Motion3D::hashPose(restored.static_initial_tcw),
        expectedHash);
}

TEST(FrozenReplayPacketTest, ReplaysFrozenLocalMapPosterior) {
    Motion3D::FrozenReplayPacket packet = syntheticPacket();
    packet.posterior_min_score_improvement = 0.0015f;
    packet.posterior_evaluated = true;
    packet.posterior_static_initial_tcw = Sophus::SE3f();
    packet.posterior_dynamic_prior_tcw = packet.dynamic_initial_tcw;
    packet.posterior_dynamic_information =
        1e6f * Motion3D::isotropicTranslationInformation(
                   Eigen::Matrix<float, 6, 6>::Identity());

    const Motion3D::FrozenReplayPosteriorResult result =
        Motion3D::runFrozenReplayPosterior(packet);

    ASSERT_TRUE(result.valid) << result.error;
    EXPECT_TRUE(result.gate_valid);
    EXPECT_EQ(result.static_branch.matches, 30);
    EXPECT_EQ(result.dynamic_branch.matches, 30);
    EXPECT_EQ(result.static_branch.inliers, 30);
    EXPECT_EQ(result.dynamic_branch.inliers, 30);
    EXPECT_EQ(result.common_support, 30);
    EXPECT_EQ(result.static_frame.map_point_ids.size(), 30u);
    EXPECT_EQ(result.dynamic_frame.map_point_ids.size(), 30u);
    EXPECT_LT(
        result.static_branch.optimized_tcw.translation().norm(), 1e-5f);
    EXPECT_GT(
        result.dynamic_branch.optimized_tcw.translation().x(), 0.001f);
    EXPECT_LT(result.static_score, result.dynamic_score);
    EXPECT_FALSE(result.pass);
}

TEST(FrozenReplayPacketTest, StaticPosteriorScanDoesNotComputeDynamicBranch) {
    Motion3D::FrozenReplayPacket packet = syntheticPacket();
    packet.posterior_evaluated = true;
    packet.posterior_static_initial_tcw = Sophus::SE3f();
    packet.posterior_dynamic_prior_tcw = packet.dynamic_initial_tcw;
    packet.posterior_dynamic_information =
        Motion3D::isotropicTranslationInformation(
            Eigen::Matrix<float, 6, 6>::Identity());

    const Motion3D::FrozenReplayPosteriorResult result =
        Motion3D::runFrozenReplayPosterior(
            packet,
            packet.posterior_static_initial_tcw,
            packet.posterior_dynamic_prior_tcw,
            packet.posterior_dynamic_information,
            1.0f,
            0.0f,
            0.025f,
            Motion3D::TranslationLeverageMode::NormalizeToTarget,
            false);

    ASSERT_TRUE(result.valid) << result.error;
    EXPECT_EQ(result.static_branch.matches, 30);
    EXPECT_EQ(result.static_branch.inliers, 30);
    EXPECT_TRUE(result.static_frame.valid);
    EXPECT_EQ(result.dynamic_branch.matches, -1);
    EXPECT_EQ(result.dynamic_branch.inliers, -1);
    EXPECT_FALSE(result.dynamic_frame.valid);
    EXPECT_FALSE(result.gate_valid);
    EXPECT_FALSE(result.pass);
}

TEST(FrozenReplayPacketTest, BuildsIsotropicTranslationInformation) {
    Eigen::Matrix<float, 6, 6> input =
        Eigen::Matrix<float, 6, 6>::Zero();
    input(0, 0) = 3.0f;
    input(1, 1) = 6.0f;
    input(2, 2) = 9.0f;
    input(0, 4) = 100.0f;

    const Eigen::Matrix<float, 6, 6> result =
        Motion3D::isotropicTranslationInformation(input);
    const Eigen::Matrix3f translation =
        result.topLeftCorner<3, 3>();
    const Eigen::Matrix3f translationRotation =
        result.topRightCorner<3, 3>();
    const Eigen::Matrix3f rotationTranslation =
        result.bottomLeftCorner<3, 3>();
    const Eigen::Matrix3f rotation =
        result.bottomRightCorner<3, 3>();

    EXPECT_TRUE(
        translation.isApprox(
            6.0f * Eigen::Matrix3f::Identity(), 0.0f));
    EXPECT_FLOAT_EQ(translationRotation.norm(), 0.0f);
    EXPECT_FLOAT_EQ(rotationTranslation.norm(), 0.0f);
    EXPECT_FLOAT_EQ(rotation.norm(), 0.0f);
}

TEST(FrozenReplayPacketTest, ReplaysSchurInformationFactor) {
    Motion3D::FrozenReplayPacket packet = syntheticPacket();
    packet.dynamic_initialization_only = false;
    packet.dynamic_information_scale = 1.0f;
    packet.dynamic_pose_information =
        1e6f * Eigen::Matrix<float, 6, 6>::Identity();

    const Motion3D::FrozenReplayResult result =
        Motion3D::runFrozenReplay(packet);

    ASSERT_TRUE(result.valid) << result.error;
    EXPECT_LT(
        result.static_branch.optimized_tcw.translation().norm(), 1e-5f);
    EXPECT_GT(
        result.dynamic_branch.optimized_tcw.translation().x(), 0.001f);
}

TEST(FrozenReplayPacketTest, ProjectsPriorToTranslationSubspace) {
    const Sophus::SE3f reference =
        Sophus::SE3f::exp(
            (Eigen::Matrix<float, 6, 1>() <<
                 0.2f, -0.1f, 0.05f, 0.03f, -0.02f, 0.04f)
                .finished());
    const Eigen::Matrix<float, 6, 1> original_increment =
        (Eigen::Matrix<float, 6, 1>() <<
             0.04f, -0.03f, 0.02f, 0.08f, -0.05f, 0.06f)
            .finished();
    Sophus::SE3f prior =
        Sophus::SE3f::exp(original_increment) * reference;
    Eigen::Matrix<float, 6, 6> factor =
        Eigen::Matrix<float, 6, 6>::Random();
    Eigen::Matrix<float, 6, 6> information =
        factor.transpose() * factor;
    const Eigen::Matrix3f translation_information =
        information.topLeftCorner<3, 3>();

    Motion3D::projectPosePriorToTranslationSubspace(
        reference, prior, information);

    const Eigen::Matrix<float, 6, 1> projected_increment =
        (prior * reference.inverse()).log();
    EXPECT_TRUE(
        projected_increment.head<3>().isApprox(
            original_increment.head<3>(), 1e-5f));
    EXPECT_LT(projected_increment.tail<3>().norm(), 1e-5f);
    const Eigen::Matrix3f projected_translation =
        information.topLeftCorner<3, 3>();
    const float translation_rotation_norm =
        information.topRightCorner<3, 3>().norm();
    const float rotation_translation_norm =
        information.bottomLeftCorner<3, 3>().norm();
    const float rotation_norm =
        information.bottomRightCorner<3, 3>().norm();
    EXPECT_TRUE(
        projected_translation.isApprox(translation_information, 1e-5f));
    EXPECT_LT(translation_rotation_norm, 1e-6f);
    EXPECT_LT(rotation_translation_norm, 1e-6f);
    EXPECT_LT(rotation_norm, 1e-6f);
}

TEST(FrozenReplayPacketTest, ProjectsCameraInterventionInWorldFrame) {
    const Sophus::SE3f static_tcw =
        Sophus::SE3f(
            Sophus::SO3f::exp(
                Eigen::Vector3f(0.1f, -0.2f, 0.05f)),
            Eigen::Vector3f(0.4f, -0.1f, 0.2f));
    const Sophus::SE3f dynamic_tcw =
        Sophus::SE3f(
            Sophus::SO3f::exp(
                Eigen::Vector3f(-0.2f, 0.05f, 0.12f)),
            Eigen::Vector3f(-0.3f, 0.25f, 0.1f));

    const Sophus::SE3f projected_twc =
        Motion3D::projectCameraInterventionToTranslation(
            static_tcw, dynamic_tcw).inverse();

    EXPECT_TRUE(
        projected_twc.translation().isApprox(
            dynamic_tcw.inverse().translation(), 1e-6f));
    EXPECT_LT(
        (projected_twc.so3().inverse() *
         static_tcw.inverse().so3()).log().norm(),
        1e-6f);
}

TEST(FrozenReplayPacketTest, ReusesShadowTranslationAndLiveRotation) {
    const Sophus::SE3f live_tcw =
        Sophus::SE3f(
            Sophus::SO3f::exp(
                Eigen::Vector3f(0.1f, -0.2f, 0.05f)),
            Eigen::Vector3f(0.4f, -0.1f, 0.2f));
    const Sophus::SE3f shadow_tcw =
        Sophus::SE3f(
            Sophus::SO3f::exp(
                Eigen::Vector3f(-0.2f, 0.05f, 0.12f)),
            Eigen::Vector3f(-0.3f, 0.25f, 0.1f));

    const Sophus::SE3f reused_twc =
        Motion3D::reuseShadowCameraTranslation(
            live_tcw, shadow_tcw).inverse();

    EXPECT_TRUE(
        reused_twc.translation().isApprox(
            shadow_tcw.inverse().translation(), 1e-6f));
    EXPECT_LT(
        (reused_twc.so3().inverse() *
         live_tcw.inverse().so3()).log().norm(),
        1e-6f);

    const Sophus::SE3f halfway_twc =
        Motion3D::reuseShadowCameraTranslation(
            live_tcw, shadow_tcw, 0.5f).inverse();
    EXPECT_TRUE(
        halfway_twc.translation().isApprox(
            0.5f * (live_tcw.inverse().translation() +
                    shadow_tcw.inverse().translation()),
            1e-6f));
    EXPECT_LT(
        (halfway_twc.so3().inverse() *
         live_tcw.inverse().so3()).log().norm(),
        1e-6f);
}

TEST(FrozenReplayPacketTest, VelocityNeutralUpdateKeepsIncomingIncrement) {
    const Sophus::SE3f incoming(
        Sophus::SO3f::exp(Eigen::Vector3f(0.01f, -0.02f, 0.03f)),
        Eigen::Vector3f(0.1f, -0.2f, 0.3f));
    const Sophus::SE3f previous(
        Sophus::SO3f::exp(Eigen::Vector3f(-0.03f, 0.02f, 0.01f)),
        Eigen::Vector3f(1.0f, 2.0f, 3.0f));
    const Sophus::SE3f current(
        Sophus::SO3f::exp(Eigen::Vector3f(0.04f, 0.01f, -0.02f)),
        Eigen::Vector3f(1.4f, 1.8f, 3.2f));

    const Sophus::SE3f neutral = Motion3D::updateCameraVelocity(
        incoming, previous, current, true);
    const Sophus::SE3f normal = Motion3D::updateCameraVelocity(
        incoming, previous, current, false);

    EXPECT_TRUE(neutral.matrix().isApprox(incoming.matrix(), 1e-6f));
    EXPECT_TRUE(normal.matrix().isApprox(
        (current * previous.inverse()).matrix(), 1e-6f));
}

TEST(FrozenReplayPacketTest, BoundsDynamicTranslationLeverage) {
    Eigen::Matrix<float, 6, 6> staticInformation =
        Eigen::Matrix<float, 6, 6>::Zero();
    staticInformation.diagonal().head<3>() <<
        100.0f, 200.0f, 400.0f;
    Eigen::Matrix<float, 6, 6> dynamicInformation =
        Eigen::Matrix<float, 6, 6>::Zero();
    dynamicInformation.diagonal().head<3>().setConstant(1000.0f);

    const Eigen::Matrix<float, 6, 6> bounded =
        Motion3D::leverageBoundedTranslationInformation(
            staticInformation, dynamicInformation, 0.25f);

    EXPECT_NEAR(bounded(0, 0), 25.0f, 1e-2f);
    EXPECT_NEAR(bounded(1, 1), 50.0f, 1e-2f);
    EXPECT_NEAR(bounded(2, 2), 100.0f, 1e-2f);
    const float rotationalInformation =
        bounded.block<3, 3>(3, 3).cwiseAbs().maxCoeff();
    EXPECT_LT(rotationalInformation, 1e-7f);
}

TEST(FrozenReplayPacketTest, PreservesDynamicInformationBelowLeverageCap) {
    Eigen::Matrix<float, 6, 6> staticInformation =
        Eigen::Matrix<float, 6, 6>::Zero();
    staticInformation.diagonal().head<3>() <<
        100.0f, 200.0f, 400.0f;
    Eigen::Matrix<float, 6, 6> dynamicInformation =
        Eigen::Matrix<float, 6, 6>::Zero();
    dynamicInformation.diagonal().head<3>() <<
        10.0f, 20.0f, 40.0f;

    const Eigen::Matrix<float, 6, 6> bounded =
        Motion3D::leverageBoundedTranslationInformation(
            staticInformation, dynamicInformation, 0.5f);

    EXPECT_NEAR(bounded(0, 0), 10.0f, 1e-2f);
    EXPECT_NEAR(bounded(1, 1), 20.0f, 1e-2f);
    EXPECT_NEAR(bounded(2, 2), 40.0f, 1e-2f);
}

TEST(FrozenReplayPacketTest, NormalizesWeakDynamicInformationToTargetLeverage) {
    Eigen::Matrix<float, 6, 6> staticInformation =
        Eigen::Matrix<float, 6, 6>::Zero();
    staticInformation.diagonal().head<3>() <<
        100.0f, 200.0f, 400.0f;
    Eigen::Matrix<float, 6, 6> dynamicInformation =
        Eigen::Matrix<float, 6, 6>::Zero();
    dynamicInformation.diagonal().head<3>() <<
        1.0f, 4.0f, 20.0f;
    Motion3D::TranslationLeverageDiagnostics diagnostics;

    const Eigen::Matrix<float, 6, 6> calibrated =
        Motion3D::leverageCalibratedTranslationInformation(
            staticInformation, dynamicInformation, 0.1f,
            Motion3D::TranslationLeverageMode::NormalizeToTarget,
            &diagnostics);

    EXPECT_NEAR(calibrated(0, 0), 2.0f, 1e-3f);
    EXPECT_NEAR(calibrated(1, 1), 8.0f, 1e-3f);
    EXPECT_NEAR(calibrated(2, 2), 40.0f, 1e-3f);
    EXPECT_TRUE(diagnostics.valid);
    EXPECT_NEAR(diagnostics.max_generalized_leverage_raw, 0.05f, 1e-4f);
    EXPECT_NEAR(
        diagnostics.max_generalized_leverage_bounded, 0.1f, 1e-4f);
    EXPECT_NEAR(diagnostics.normalization_scale, 2.0f, 1e-4f);
}

TEST(FrozenReplayPacketTest, NormalizesStrongDynamicInformationToTargetLeverage) {
    Eigen::Matrix<float, 6, 6> staticInformation =
        Eigen::Matrix<float, 6, 6>::Zero();
    staticInformation.diagonal().head<3>() <<
        100.0f, 200.0f, 400.0f;
    Eigen::Matrix<float, 6, 6> dynamicInformation =
        Eigen::Matrix<float, 6, 6>::Zero();
    dynamicInformation.diagonal().head<3>().setConstant(100.0f);
    Motion3D::TranslationLeverageDiagnostics diagnostics;

    const Eigen::Matrix<float, 6, 6> calibrated =
        Motion3D::leverageCalibratedTranslationInformation(
            staticInformation, dynamicInformation, 0.1f,
            Motion3D::TranslationLeverageMode::NormalizeToTarget,
            &diagnostics);

    EXPECT_NEAR(calibrated(0, 0), 10.0f, 1e-3f);
    EXPECT_NEAR(calibrated(1, 1), 10.0f, 1e-3f);
    EXPECT_NEAR(calibrated(2, 2), 10.0f, 1e-3f);
    EXPECT_NEAR(diagnostics.max_generalized_leverage_raw, 1.0f, 1e-4f);
    EXPECT_NEAR(
        diagnostics.max_generalized_leverage_bounded, 0.1f, 1e-4f);
    EXPECT_NEAR(diagnostics.normalization_scale, 0.1f, 1e-4f);
}

TEST(FrozenReplayPacketTest, UsesRotationMarginalizedStaticTranslationInformation) {
    Eigen::Matrix<float, 6, 6> staticInformation =
        Eigen::Matrix<float, 6, 6>::Zero();
    staticInformation.diagonal().head<3>() <<
        100.0f, 200.0f, 400.0f;
    staticInformation.diagonal().tail<3>().setConstant(100.0f);
    staticInformation(0, 3) = 90.0f;
    staticInformation(3, 0) = 90.0f;
    Eigen::Matrix<float, 6, 6> dynamicInformation =
        Eigen::Matrix<float, 6, 6>::Zero();
    dynamicInformation.diagonal().head<3>().setConstant(1000.0f);
    Motion3D::TranslationLeverageDiagnostics diagnostics;

    const Eigen::Matrix<float, 6, 6> bounded =
        Motion3D::leverageBoundedTranslationInformation(
            staticInformation, dynamicInformation, 0.25f,
            &diagnostics);

    EXPECT_NEAR(bounded(0, 0), 4.75f, 1e-2f);
    EXPECT_NEAR(bounded(1, 1), 50.0f, 1e-2f);
    EXPECT_NEAR(bounded(2, 2), 100.0f, 1e-2f);
    EXPECT_TRUE(diagnostics.valid);
    EXPECT_NEAR(diagnostics.static_translation_trace, 619.0f, 1e-2f);
    EXPECT_LE(diagnostics.max_generalized_leverage_bounded, 0.2501f);
}

TEST(FrozenReplayPacketTest, RejectsInvalidLeverageInputs) {
    const Eigen::Matrix<float, 6, 6> information =
        Eigen::Matrix<float, 6, 6>::Identity();
    EXPECT_TRUE(
        Motion3D::leverageBoundedTranslationInformation(
            information, information, 0.0f).isZero());
    EXPECT_TRUE(
        Motion3D::leverageBoundedTranslationInformation(
            information, information,
            std::numeric_limits<float>::quiet_NaN()).isZero());
}

TEST(FrozenReplayPacketTest, PosteriorGateUsesInliersAndMatchedCost) {
    EXPECT_TRUE(Motion3D::posteriorLocalMapGate(
        40, 41, 2.0f, 2.0001f, 0.0015f));
    EXPECT_FALSE(Motion3D::posteriorLocalMapGate(
        40, 41, 2.0f, 2.01f, 0.0015f));
    EXPECT_TRUE(Motion3D::posteriorLocalMapGate(
        40, 40, 2.0f, 1.99f, 0.0015f));
    EXPECT_FALSE(Motion3D::posteriorLocalMapGate(
        40, 40, 2.0f, 1.999f, 0.0015f));
    EXPECT_TRUE(Motion3D::posteriorLocalMapGate(
        40, 40, 2.0f, 1.998f, 0.0015f));
    EXPECT_FALSE(Motion3D::posteriorLocalMapGate(
        -1, 40, 2.0f, 1.0f, 0.0015f));
    EXPECT_FALSE(Motion3D::posteriorLocalMapGate(
        40, 40, 2.0f, 1.0f, -0.1f));
}

TEST(FrozenReplayPacketTest, MeasuresCameraCenterTranslationInnovation) {
    const Sophus::SE3f first_twc(
        Sophus::SO3f::exp(Eigen::Vector3f(0.1f, -0.2f, 0.05f)),
        Eigen::Vector3f(0.4f, -0.1f, 0.2f));
    const Sophus::SE3f second_twc(
        Sophus::SO3f::exp(Eigen::Vector3f(-0.2f, 0.05f, 0.12f)),
        Eigen::Vector3f(0.403f, -0.104f, 0.2f));

    EXPECT_NEAR(
        Motion3D::cameraCenterTranslationInnovation(
            first_twc.inverse(), second_twc.inverse()),
        0.005f, 1e-6f);
}

TEST(FrozenReplayPacketTest, ExplicitBypassSkipsAllReliabilityGates) {
    Motion3D::FrozenReplayPacket packet = syntheticPacket();
    packet.gate.min_inlier_gain = 1000;
    packet.gate.use_direct_validation = true;
    packet.gate.require_common_support_improvement = true;
    packet.gate.bypass_reliability_gate = true;

    const Motion3D::FrozenReplayResult result =
        Motion3D::runFrozenReplay(packet);

    ASSERT_TRUE(result.valid) << result.error;
    EXPECT_TRUE(result.consensus_pass);
    EXPECT_TRUE(result.direct_pass);
    EXPECT_TRUE(result.common_support_pass);
    EXPECT_TRUE(result.would_use);
}

TEST(FrozenReplayPacketTest, CarriesBranchStateAcrossPackets) {
    Motion3D::FrozenReplayPacket firstPacket = syntheticPacket();
    const Motion3D::FrozenReplayBranchStateResult first =
        Motion3D::runFrozenReplayBranch(
            firstPacket, Motion3D::FrozenReplayFrameState(),
            firstPacket.dynamic_initial_tcw);
    ASSERT_TRUE(first.valid) << first.error;
    ASSERT_TRUE(first.frame.valid);
    EXPECT_EQ(first.branch.matches, 30);
    EXPECT_EQ(first.branch.inliers, 30);

    Motion3D::FrozenReplayPacket secondPacket = firstPacket;
    secondPacket.previous_timestamp = firstPacket.current_timestamp;
    secondPacket.current_timestamp =
        firstPacket.current_timestamp + 0.033;
    secondPacket.previous_keypoints = firstPacket.current_keypoints;
    secondPacket.previous_feature_count = firstPacket.feature_count;
    secondPacket.previous_tcw = first.branch.optimized_tcw;
    const Sophus::SE3f relative =
        first.branch.optimized_tcw *
        firstPacket.previous_tcw.inverse();
    const Sophus::SE3f initial =
        relative * first.branch.optimized_tcw;
    const Motion3D::FrozenReplayBranchStateResult second =
        Motion3D::runFrozenReplayBranch(
            secondPacket, first.frame, initial);
    ASSERT_TRUE(second.valid) << second.error;
    EXPECT_EQ(second.branch.matches, 30);
    EXPECT_EQ(second.branch.inliers, 30);
    EXPECT_EQ(second.frame.map_point_ids.size(), 30u);
}
