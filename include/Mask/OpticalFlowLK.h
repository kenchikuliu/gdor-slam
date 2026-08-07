/**
 * This file is part of DyGeoFusion-SLAM+
 *
 * Copyright (C) 2024 DyGeoFusion-SLAM+ Authors.
 *
 * DyGeoFusion-SLAM+ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>

namespace DyGeoFusion
{

/**
 * @brief OpticalFlowLK - Lucas-Kanade optical flow for motion detection
 *
 * Implements sparse LK optical flow on a grid of points to detect
 * inter-frame motion. The flow magnitudes are used to identify
 * dynamic regions in the scene.
 */
class OpticalFlowLK
{
public:
    /**
     * @brief Construct OpticalFlowLK calculator
     * @param win_size LK window size
     * @param max_level Maximum pyramid level
     * @param grid_step Grid step for sparse point sampling
     * @param flow_threshold Flow magnitude threshold for motion detection
     */
    OpticalFlowLK(int win_size = 21,
                  int max_level = 3,
                  int grid_step = 8,
                  float flow_threshold = 1.5f);

    /**
     * @brief Destructor
     */
    ~OpticalFlowLK() = default;

    /**
     * @brief Compute optical flow between two frames
     * @param prev_gray Previous frame (grayscale)
     * @param curr_gray Current frame (grayscale)
     * @param prev_points Previous frame points (optional, will generate if empty)
     * @return Flow vectors for each point
     */
    std::vector<cv::Point2f> compute(const cv::Mat& prev_gray,
                                      const cv::Mat& curr_gray,
                                      const std::vector<cv::Point2f>& prev_points = {});

    /**
     * @brief Generate motion mask from optical flow
     * @param prev_gray Previous frame (grayscale)
     * @param curr_gray Current frame (grayscale)
     * @param roi Optional ROI mask to restrict flow computation
     * @return cv::Mat Motion mask (CV_8UC1, 1=motion detected, 0=static)
     */
    cv::Mat getMotionMask(const cv::Mat& prev_gray,
                          const cv::Mat& curr_gray,
                          const cv::Mat& roi = cv::Mat());

    /**
     * @brief Motion mask from flow residuals after robust global ego-motion compensation.
     *
     * A RANSAC affine model is fitted to the dominant image motion. Only LK
     * residuals relative to that model are classified as independent motion.
     */
    cv::Mat getResidualMotionMask(const cv::Mat& prev_gray,
                                  const cv::Mat& curr_gray,
                                  const cv::Mat& roi = cv::Mat());

    /**
     * @brief Evaluate residual motion over the full image while fitting the
     * dominant camera-motion model only on a trusted static ROI.
     */
    cv::Mat getFullFrameResidualMotionMask(
        const cv::Mat& prev_gray,
        const cv::Mat& curr_gray,
        const cv::Mat& static_model_roi,
        float high_confidence_scale = 1.0f);

    bool lastResidualModelValid() const
    {
        return last_residual_model_valid_;
    }

    const cv::Mat& lastStaticResidualMotionMask() const
    {
        return last_static_residual_mask_;
    }

    const cv::Mat& lastHighConfidenceResidualMotionMask() const
    {
        return last_high_confidence_residual_mask_;
    }

    /**
     * @brief Get dense flow visualization
     * @param flow_vectors Flow vectors
     * @param prev_points Previous points
     * @param image_size Output image size
     * @return cv::Mat Visualization image (CV_8UC3)
     */
    cv::Mat visualizeFlow(const std::vector<cv::Point2f>& flow_vectors,
                          const std::vector<cv::Point2f>& prev_points,
                          const cv::Size& image_size) const;

    /**
     * @brief Generate grid points for sparse flow
     * @param image_size Image size
     * @param step Grid step
     * @param mask Optional mask to filter points
     * @return Vector of grid points
     */
    static std::vector<cv::Point2f> generateGridPoints(
        const cv::Size& image_size,
        int step,
        const cv::Mat& mask = cv::Mat());

    /**
     * @brief Set flow threshold
     * @param threshold New threshold in pixels
     */
    void setFlowThreshold(float threshold) { flow_threshold_ = threshold; }

    /**
     * @brief Get flow threshold
     */
    float getFlowThreshold() const { return flow_threshold_; }

    /**
     * @brief Get last computed flow vectors
     */
    const std::vector<cv::Point2f>& getLastFlowVectors() const { return last_flow_; }

    /**
     * @brief Get last tracked points
     */
    const std::vector<cv::Point2f>& getLastTrackedPoints() const { return last_tracked_; }

    /**
     * @brief Get last computation time in milliseconds
     */
    float getLastComputeTime() const { return last_compute_time_ms_; }

private:
    /**
     * @brief Interpolate sparse flow to dense motion mask
     * @param prev_points Source points
     * @param flow_vectors Flow vectors
     * @param status Tracking status
     * @param image_size Output size
     * @return Dense motion mask
     */
    cv::Mat interpolateToMask(const std::vector<cv::Point2f>& prev_points,
                              const std::vector<cv::Point2f>& flow_vectors,
                              const std::vector<uchar>& status,
                              const cv::Size& image_size,
                              float threshold = -1.0f);

private:
    int win_size_;
    int max_level_;
    int grid_step_;
    float flow_threshold_;
    float last_compute_time_ms_;
    bool last_residual_model_valid_;
    cv::Mat last_static_residual_mask_;
    cv::Mat last_high_confidence_residual_mask_;

    // LK parameters
    cv::TermCriteria term_criteria_;

    // Cached results
    std::vector<cv::Point2f> last_flow_;
    std::vector<cv::Point2f> last_tracked_;
    std::vector<uchar> last_status_;
};

} // namespace DyGeoFusion
