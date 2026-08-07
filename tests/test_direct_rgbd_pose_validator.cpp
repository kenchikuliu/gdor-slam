#include <gtest/gtest.h>

#include "Motion3D/DirectRgbdPoseValidator.h"
#include "Motion3D/ReplayStateContract.h"

namespace {

Motion3D::DirectRgbdPoseValidator::Intrinsics intrinsics() {
    return {80.0f, 80.0f, 31.5f, 23.5f};
}

cv::Mat gradientImage() {
    cv::Mat image(48, 64, CV_8UC1);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            image.at<unsigned char>(y, x) =
                static_cast<unsigned char>(2 * x + y);
        }
    }
    return image;
}

}  // namespace

TEST(DirectRgbdPoseValidatorTest, IdentityBeatsIncorrectForwardMotion) {
    Motion3D::DirectRgbdPoseValidator::Config config;
    config.grid_step = 4;
    config.min_common_support = 20;
    Motion3D::DirectRgbdPoseValidator validator(config);
    const cv::Mat image = gradientImage();
    const cv::Mat depth(image.size(), CV_32FC1, cv::Scalar(2.0f));
    const cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
    const Sophus::SE3f identity;
    Sophus::SE3f wrong;
    wrong.translation().z() = 0.1f;

    const auto result = validator.score(
        image, image, depth, depth, mask, mask,
        identity, identity, wrong, intrinsics());

    ASSERT_TRUE(result.valid);
    EXPECT_GT(result.common_support, 100);
    EXPECT_LT(result.static_depth_score, result.dynamic_depth_score);
    EXPECT_LT(result.static_photometric_score, result.dynamic_photometric_score);
    EXPECT_LT(result.static_combined_score, result.dynamic_combined_score);
}

TEST(DirectRgbdPoseValidatorTest, UsesOnlyCommonStaticSupport) {
    Motion3D::DirectRgbdPoseValidator::Config config;
    config.grid_step = 4;
    config.min_common_support = 20;
    Motion3D::DirectRgbdPoseValidator validator(config);
    const cv::Mat image = gradientImage();
    const cv::Mat depth(image.size(), CV_32FC1, cv::Scalar(2.0f));
    cv::Mat previous_mask(image.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat current_mask(image.size(), CV_8UC1, cv::Scalar(255));
    current_mask.colRange(0, image.cols / 2).setTo(0);
    const Sophus::SE3f identity;

    const auto full = validator.score(
        image, image, depth, depth, previous_mask,
        cv::Mat(image.size(), CV_8UC1, cv::Scalar(255)),
        identity, identity, identity, intrinsics());
    const auto masked = validator.score(
        image, image, depth, depth, previous_mask, current_mask,
        identity, identity, identity, intrinsics());

    ASSERT_TRUE(full.valid);
    ASSERT_TRUE(masked.valid);
    EXPECT_LT(masked.common_support, full.common_support);
    EXPECT_NEAR(masked.static_combined_score, 0.0f, 1e-6f);
    EXPECT_NEAR(masked.dynamic_combined_score, 0.0f, 1e-6f);
}

TEST(DirectRgbdPoseValidatorTest, SelectsConfiguredScoreMode) {
    Motion3D::DirectRgbdPoseValidator::Result result;
    result.valid = true;
    result.static_photometric_score = 1.0f;
    result.dynamic_photometric_score = 2.0f;
    result.static_depth_score = 2.0f;
    result.dynamic_depth_score = 1.0f;
    result.static_combined_score = 2.0f;
    result.dynamic_combined_score = 1.5f;

    EXPECT_FALSE(Motion3D::DirectRgbdPoseValidator::prefersDynamic(
        result,
        Motion3D::DirectRgbdPoseValidator::ScoreMode::Photometric));
    EXPECT_TRUE(Motion3D::DirectRgbdPoseValidator::prefersDynamic(
        result, Motion3D::DirectRgbdPoseValidator::ScoreMode::Depth));
    EXPECT_TRUE(Motion3D::DirectRgbdPoseValidator::prefersDynamic(
        result, Motion3D::DirectRgbdPoseValidator::ScoreMode::Combined));

    result.valid = false;
    EXPECT_FALSE(Motion3D::DirectRgbdPoseValidator::prefersDynamic(
        result, Motion3D::DirectRgbdPoseValidator::ScoreMode::Combined));
}

TEST(
    DirectRgbdPoseValidatorTest,
    RefreshedPreviousPoseSurvivesReferenceKeyframeBaRewrite) {
    Motion3D::DirectRgbdPoseValidator::Config config;
    config.grid_step = 4;
    config.min_common_support = 20;
    Motion3D::DirectRgbdPoseValidator validator(config);
    const cv::Mat image = gradientImage();
    const cv::Mat depth(image.size(), CV_32FC1, cv::Scalar(2.0f));
    const cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));

    Sophus::SE3f frame_from_reference;
    frame_from_reference.translation().y() = 0.01f;
    Sophus::SE3f reference_before_ba;
    Sophus::SE3f reference_after_ba;
    reference_after_ba.translation().x() = 0.05f;
    const Sophus::SE3f stale_previous =
        Motion3D::canonicalPreviousTcw(
            frame_from_reference, reference_before_ba);
    const Sophus::SE3f refreshed_previous =
        Motion3D::canonicalPreviousTcw(
            frame_from_reference, reference_after_ba);
    const Sophus::SE3f current_in_post_ba_map = refreshed_previous;

    const auto refreshed = validator.score(
        image, image, depth, depth, mask, mask,
        refreshed_previous, current_in_post_ba_map,
        current_in_post_ba_map, intrinsics());
    const auto stale = validator.score(
        image, image, depth, depth, mask, mask,
        stale_previous, current_in_post_ba_map,
        current_in_post_ba_map, intrinsics());

    ASSERT_TRUE(refreshed.valid);
    ASSERT_TRUE(stale.valid);
    EXPECT_NEAR(refreshed.static_combined_score, 0.0f, 1e-6f);
    EXPECT_GT(stale.static_combined_score, refreshed.static_combined_score);
    EXPECT_GT(stale.static_photometric_score, 0.0f);
}

TEST(ReplayStateContractTest, HashesCanonicalSortedState) {
    Motion3D::ReplayMapPointState first{
        4, Eigen::Vector3f(1.0f, 2.0f, 3.0f)};
    Motion3D::ReplayMapPointState second{
        9, Eigen::Vector3f(-1.0f, 0.5f, 2.5f)};
    EXPECT_EQ(
        Motion3D::hashMapPointStates({first, second}),
        Motion3D::hashMapPointStates({second, first}));

    second.world_position.z() += 0.001f;
    EXPECT_NE(
        Motion3D::hashMapPointStates({first, second}),
        Motion3D::hashMapPointStates({
            first,
            Motion3D::ReplayMapPointState{
                9, Eigen::Vector3f(-1.0f, 0.5f, 2.5f)}}));

    Motion3D::ReplayMapPointTrackingState trackingFirst;
    trackingFirst.map_point_id = 4;
    trackingFirst.world_position = first.world_position;
    trackingFirst.normal = Eigen::Vector3f::UnitZ();
    trackingFirst.min_distance = 0.5f;
    trackingFirst.max_distance = 5.0f;
    trackingFirst.descriptor_hash = 11;
    trackingFirst.observations = 3;
    Motion3D::ReplayMapPointTrackingState trackingSecond = trackingFirst;
    trackingSecond.map_point_id = 9;
    const Motion3D::ReplayMapPointTrackingDigest initialDigest =
        Motion3D::digestMapPointTrackingStates(
            {trackingFirst, trackingSecond});
    EXPECT_EQ(
        Motion3D::hashMapPointTrackingStates(
            {trackingFirst, trackingSecond}),
        Motion3D::hashMapPointTrackingStates(
            {trackingSecond, trackingFirst}));
    trackingSecond.descriptor_hash = 12;
    EXPECT_NE(
        Motion3D::hashMapPointTrackingStates(
            {trackingFirst, trackingSecond}),
        Motion3D::hashMapPointTrackingStates({
            trackingFirst,
            Motion3D::ReplayMapPointTrackingState{
                9, first.world_position, Eigen::Vector3f::UnitZ(),
                0.5f, 5.0f, 11, 3}}));
    const Motion3D::ReplayMapPointTrackingDigest changedDigest =
        Motion3D::digestMapPointTrackingStates(
            {trackingFirst, trackingSecond});
    EXPECT_EQ(initialDigest.id_hash, changedDigest.id_hash);
    EXPECT_EQ(initialDigest.position_hash, changedDigest.position_hash);
    EXPECT_EQ(initialDigest.normal_hash, changedDigest.normal_hash);
    EXPECT_EQ(initialDigest.distance_hash, changedDigest.distance_hash);
    EXPECT_NE(initialDigest.descriptor_hash, changedDigest.descriptor_hash);
    EXPECT_EQ(initialDigest.observation_hash, changedDigest.observation_hash);
    EXPECT_NE(
        Motion3D::hashIdSequence({1, 2}),
        Motion3D::hashIdSequence({2, 1}));
}

TEST(ReplayStateContractTest, HashesPoseAndSupportChanges) {
    Motion3D::ReplaySupportObservation observation;
    observation.feature_index = 3;
    observation.map_point_id = 17;
    observation.world_position = Eigen::Vector3f(1.0f, 0.0f, 2.0f);
    observation.keypoint_x = 20.0f;
    observation.keypoint_y = 10.0f;
    observation.right_coordinate = 18.0f;
    observation.octave = 2;
    const std::uint64_t initialSupport =
        Motion3D::hashSupportObservations({observation});
    observation.keypoint_x += 0.25f;
    EXPECT_NE(
        initialSupport,
        Motion3D::hashSupportObservations({observation}));

    Sophus::SE3f initialPose;
    Sophus::SE3f changedPose;
    changedPose.translation().z() = 0.01f;
    EXPECT_NE(
        Motion3D::hashPose(initialPose),
        Motion3D::hashPose(changedPose));
    EXPECT_EQ(Motion3D::hashToHex(0x2aULL), "000000000000002a");
}
