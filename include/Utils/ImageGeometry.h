#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace DyGeoFusion {

inline bool resizeRgbdToCameraSize(
    cv::Mat& rgb, cv::Mat& depth, const cv::Size& camera_size)
{
    if (rgb.empty() || depth.empty() ||
        camera_size.width <= 0 || camera_size.height <= 0) {
        return false;
    }
    if (rgb.size() != camera_size) {
        cv::resize(
            rgb, rgb, camera_size, 0.0, 0.0, cv::INTER_LINEAR);
    }
    if (depth.size() != camera_size) {
        cv::resize(
            depth, depth, camera_size, 0.0, 0.0, cv::INTER_NEAREST);
    }
    return true;
}

}  // namespace DyGeoFusion
