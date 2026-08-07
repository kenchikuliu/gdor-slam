#include <gtest/gtest.h>

#include "Mask/OpticalFlowLK.h"

#include <opencv2/imgproc.hpp>

namespace {

cv::Mat makeTexture(int rows, int cols, int seed)
{
    cv::Mat image(rows, cols, CV_8UC1);
    cv::RNG rng(seed);
    rng.fill(image, cv::RNG::UNIFORM, 0, 255);
    cv::GaussianBlur(image, image, cv::Size(3, 3), 0.5);
    return image;
}

cv::Mat translate(const cv::Mat& image, float x, float y)
{
    cv::Mat result;
    const cv::Mat transform = (cv::Mat_<double>(2, 3) << 1.0, 0.0, x,
                                                        0.0, 1.0, y);
    cv::warpAffine(image, result, transform, image.size(), cv::INTER_LINEAR,
                   cv::BORDER_REFLECT101);
    return result;
}

}  // namespace

TEST(OpticalFlowLKTest, CompensatesDominantCameraMotion)
{
    const cv::Mat previous = makeTexture(120, 160, 3);
    const cv::Mat current = translate(previous, 3.0f, 2.0f);
    DyGeoFusion::OpticalFlowLK flow(21, 3, 6, 1.5f);

    const cv::Mat residual_mask = flow.getResidualMotionMask(previous, current);
    const double dynamic_ratio = static_cast<double>(cv::countNonZero(residual_mask)) /
                                 static_cast<double>(residual_mask.total());
    EXPECT_LT(dynamic_ratio, 0.05);
}

TEST(OpticalFlowLKTest, DetectsIndependentResidualMotion)
{
    cv::Mat previous = makeTexture(120, 160, 5);
    cv::Mat current = translate(makeTexture(120, 160, 5), 3.0f, 2.0f);
    const cv::Mat object = makeTexture(48, 48, 17);
    const cv::Rect previous_box(45, 35, object.cols, object.rows);
    const cv::Rect current_box(previous_box.x + 11, previous_box.y + 2,
                               object.cols, object.rows);
    object.copyTo(previous(previous_box));
    object.copyTo(current(current_box));

    DyGeoFusion::OpticalFlowLK flow(21, 3, 6, 1.5f);
    const cv::Mat residual_mask = flow.getResidualMotionMask(previous, current);

    EXPECT_GT(cv::countNonZero(residual_mask(current_box)), 50);
    cv::Mat outside = cv::Mat::ones(residual_mask.size(), CV_8UC1);
    outside(current_box).setTo(0);
    cv::Mat outside_motion;
    cv::bitwise_and(residual_mask, outside, outside_motion);
    EXPECT_LT(cv::countNonZero(outside_motion), 1000);
}

TEST(OpticalFlowLKTest, FitsOnStaticRoiAndClassifiesMaskedRegions)
{
    cv::Mat previous = makeTexture(120, 160, 23);
    cv::Mat current = translate(previous, 3.0f, 2.0f);
    const cv::Mat object = makeTexture(42, 42, 29);
    const cv::Rect previous_box(42, 36, object.cols, object.rows);
    const cv::Rect current_box(previous_box.x + 12, previous_box.y + 2,
                               object.cols, object.rows);
    object.copyTo(previous(previous_box));
    object.copyTo(current(current_box));

    const cv::Mat missed_object = makeTexture(24, 24, 31);
    const cv::Rect missed_previous_box(
        14, 82, missed_object.cols, missed_object.rows);
    const cv::Rect missed_current_box(
        missed_previous_box.x + 10, missed_previous_box.y + 2,
        missed_object.cols, missed_object.rows);
    missed_object.copyTo(previous(missed_previous_box));
    missed_object.copyTo(current(missed_current_box));

    cv::Mat static_model_roi =
        cv::Mat::ones(previous.size(), CV_8UC1);
    const cv::Rect excluded_motion_region =
        previous_box | current_box;
    static_model_roi(excluded_motion_region).setTo(0);
    const cv::Rect excluded_static_region(108, 20, 30, 32);
    static_model_roi(excluded_static_region).setTo(0);

    DyGeoFusion::OpticalFlowLK flow(21, 3, 6, 1.5f);
    cv::theRNG().state = 41;
    const cv::Mat residual_mask =
        flow.getFullFrameResidualMotionMask(
            previous, current, static_model_roi, 2.0f);

    EXPECT_TRUE(flow.lastResidualModelValid());
    EXPECT_GT(cv::countNonZero(residual_mask(current_box)), 50);
    EXPECT_GT(
        cv::countNonZero(residual_mask(missed_current_box)), 10);

    DyGeoFusion::OpticalFlowLK raw_flow(21, 3, 6, 1.5f);
    cv::theRNG().state = 41;
    const cv::Mat raw_residual_mask =
        raw_flow.getResidualMotionMask(
            previous, current, static_model_roi);
    EXPECT_GT(
        cv::countNonZero(raw_residual_mask(missed_current_box)), 10);
    cv::Mat raw_mask_difference;
    cv::bitwise_xor(
        flow.lastStaticResidualMotionMask(),
        raw_residual_mask,
        raw_mask_difference);
    EXPECT_EQ(cv::countNonZero(raw_mask_difference), 0);
    const cv::Rect shifted_static_region(
        excluded_static_region.x + 3,
        excluded_static_region.y + 2,
        excluded_static_region.width,
        excluded_static_region.height);
    EXPECT_LT(
        cv::countNonZero(residual_mask(shifted_static_region)), 100);
    const cv::Mat& high_confidence_mask =
        flow.lastHighConfidenceResidualMotionMask();
    EXPECT_EQ(high_confidence_mask.size(), residual_mask.size());
    EXPECT_GT(cv::countNonZero(high_confidence_mask(current_box)), 10);
    EXPECT_LE(
        cv::countNonZero(high_confidence_mask),
        cv::countNonZero(residual_mask));
}
