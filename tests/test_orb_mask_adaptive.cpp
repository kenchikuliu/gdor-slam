#include <gtest/gtest.h>

#include "ORBextractor.h"

namespace {

cv::Mat checkerboardImage(int rows, int cols, int cellSize) {
    cv::Mat image(rows, cols, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < rows; y += cellSize) {
        for (int x = 0; x < cols; x += cellSize) {
            if (((x / cellSize) + (y / cellSize)) % 2 == 0) {
                image(cv::Rect(
                    x, y, std::min(cellSize, cols - x),
                    std::min(cellSize, rows - y))).setTo(255);
            }
        }
    }
    return image;
}

TEST(OrbMaskAdaptiveTest, ReallocatesFeatureBudgetInsideStaticMask) {
    const cv::Mat image = checkerboardImage(480, 640, 12);
    cv::Mat static_mask = cv::Mat::zeros(image.size(), CV_8UC1);
    static_mask(cv::Rect(0, 0, image.cols / 2, image.rows)).setTo(1);
    std::vector<int> lapping = {0, 0};

    ORB_SLAM3::ORBextractor standard(500, 1.2f, 8, 20, 7);
    std::vector<cv::KeyPoint> standard_keypoints;
    cv::Mat standard_descriptors;
    standard(
        image, static_mask, standard_keypoints,
        standard_descriptors, lapping);
    EXPECT_NEAR(standard.GetLastStaticMaskRatio(), 0.5f, 1e-6f);
    EXPECT_EQ(standard.GetLastAdaptiveFastThreshold(), 20);

    ORB_SLAM3::ORBextractor adaptive(500, 1.2f, 8, 20, 7);
    adaptive.SetMaskAdaptiveExtraction(true, 0.5f);
    std::vector<cv::KeyPoint> adaptive_keypoints;
    cv::Mat adaptive_descriptors;
    adaptive(
        image, static_mask, adaptive_keypoints,
        adaptive_descriptors, lapping);

    ASSERT_FALSE(standard_keypoints.empty());
    EXPECT_GT(adaptive_keypoints.size(), standard_keypoints.size());
    EXPECT_NEAR(adaptive.GetLastStaticMaskRatio(), 0.5f, 1e-6f);
    EXPECT_LT(adaptive.GetLastAdaptiveFastThreshold(), 20);
    for (const cv::KeyPoint& keypoint : adaptive_keypoints) {
        ASSERT_GE(keypoint.pt.x, 0.0f);
        ASSERT_LT(keypoint.pt.x, static_cast<float>(static_mask.cols));
        ASSERT_GE(keypoint.pt.y, 0.0f);
        ASSERT_LT(keypoint.pt.y, static_cast<float>(static_mask.rows));
        EXPECT_NE(
            static_mask.at<uchar>(
                cvRound(keypoint.pt.y), cvRound(keypoint.pt.x)),
            0);
    }
}

}  // namespace
