#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace DyGeoFusion {

inline cv::Mat openCvBgrToRgb(const cv::Mat& bgr)
{
    if (bgr.empty()) return cv::Mat();
    if (bgr.channels() != 3) {
        throw std::invalid_argument("openCvBgrToRgb expects a 3-channel image");
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

inline cv::Mat rgbToOpenCvBgr(const cv::Mat& rgb)
{
    if (rgb.empty()) return cv::Mat();
    if (rgb.channels() != 3) {
        throw std::invalid_argument("rgbToOpenCvBgr expects a 3-channel image");
    }
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
}

}  // namespace DyGeoFusion
