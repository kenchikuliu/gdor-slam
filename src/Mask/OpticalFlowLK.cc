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

#include "include/Mask/OpticalFlowLK.h"
#include <opencv2/calib3d.hpp>
#include <chrono>
#include <cmath>

namespace DyGeoFusion
{

OpticalFlowLK::OpticalFlowLK(int win_size,
                             int max_level,
                             int grid_step,
                             float flow_threshold)
    : win_size_(win_size),
      max_level_(max_level),
      grid_step_(grid_step),
      flow_threshold_(flow_threshold),
      last_compute_time_ms_(0.0f),
      last_residual_model_valid_(false),
      term_criteria_(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01)
{
}

std::vector<cv::Point2f> OpticalFlowLK::compute(const cv::Mat& prev_gray,
                                                 const cv::Mat& curr_gray,
                                                 const std::vector<cv::Point2f>& prev_points)
{
    std::vector<cv::Point2f> flow_vectors;

    if (prev_gray.empty() || curr_gray.empty()) {
        return flow_vectors;
    }

    auto start_time = std::chrono::steady_clock::now();

    // Generate grid points if not provided
    std::vector<cv::Point2f> points_to_track;
    if (prev_points.empty()) {
        points_to_track = generateGridPoints(prev_gray.size(), grid_step_);
    } else {
        points_to_track = prev_points;
    }

    if (points_to_track.empty()) {
        return flow_vectors;
    }

    // Track points using Lucas-Kanade
    std::vector<cv::Point2f> next_points;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        points_to_track,
        next_points,
        status,
        err,
        cv::Size(win_size_, win_size_),
        max_level_,
        term_criteria_);

    // Compute flow vectors
    flow_vectors.resize(points_to_track.size());
    for (size_t i = 0; i < points_to_track.size(); ++i) {
        if (status[i]) {
            flow_vectors[i] = next_points[i] - points_to_track[i];
        } else {
            flow_vectors[i] = cv::Point2f(0, 0);
        }
    }

    // Cache results
    last_flow_ = flow_vectors;
    last_tracked_ = next_points;
    last_status_ = status;

    auto end_time = std::chrono::steady_clock::now();
    last_compute_time_ms_ = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count() / 1000.0f;

    return flow_vectors;
}

cv::Mat OpticalFlowLK::getMotionMask(const cv::Mat& prev_gray,
                                      const cv::Mat& curr_gray,
                                      const cv::Mat& roi)
{
    cv::Mat mask = cv::Mat::zeros(curr_gray.size(), CV_8UC1);

    if (prev_gray.empty() || curr_gray.empty()) {
        return mask;
    }

    // Generate grid points (optionally filtered by ROI)
    std::vector<cv::Point2f> points;
    if (!roi.empty()) {
        points = generateGridPoints(prev_gray.size(), grid_step_, roi);
    } else {
        points = generateGridPoints(prev_gray.size(), grid_step_);
    }

    if (points.empty()) {
        return mask;
    }

    // Compute flow
    auto flow_vectors = compute(prev_gray, curr_gray, points);

    // Interpolate to dense mask
    mask = interpolateToMask(points, flow_vectors, last_status_, curr_gray.size());

    return mask;
}

cv::Mat OpticalFlowLK::getResidualMotionMask(const cv::Mat& prev_gray,
                                             const cv::Mat& curr_gray,
                                             const cv::Mat& roi)
{
    last_residual_model_valid_ = false;
    cv::Mat mask = cv::Mat::zeros(curr_gray.size(), CV_8UC1);
    last_static_residual_mask_ = mask.clone();
    last_high_confidence_residual_mask_ = mask.clone();
    if (prev_gray.empty() || curr_gray.empty() || prev_gray.size() != curr_gray.size()) {
        return mask;
    }

    const std::vector<cv::Point2f> points = generateGridPoints(
        prev_gray.size(), grid_step_, roi);
    if (points.size() < 6) return mask;

    compute(prev_gray, curr_gray, points);
    std::vector<cv::Point2f> backward_points;
    std::vector<uchar> backward_status;
    std::vector<float> backward_error;
    cv::calcOpticalFlowPyrLK(
        curr_gray, prev_gray, last_tracked_, backward_points,
        backward_status, backward_error, cv::Size(win_size_, win_size_),
        max_level_, term_criteria_);

    std::vector<cv::Point2f> valid_previous;
    std::vector<cv::Point2f> valid_current;
    std::vector<int> valid_indices;
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (!last_status_[i] || !backward_status[i] ||
            cv::norm(points[i] - backward_points[i]) > 1.5f) {
            continue;
        }
        valid_previous.push_back(points[i]);
        valid_current.push_back(last_tracked_[i]);
        valid_indices.push_back(static_cast<int>(i));
    }
    if (valid_previous.size() < 6) return mask;

    cv::Mat affine_inliers;
    const cv::Mat affine = cv::estimateAffinePartial2D(
        valid_previous, valid_current, affine_inliers, cv::RANSAC,
        std::max(1.0, static_cast<double>(flow_threshold_)), 2000, 0.99, 10);
    if (affine.empty()) return mask;

    cv::Mat fundamental_inliers;
    const cv::Mat fundamental = cv::findFundamentalMat(
        valid_previous, valid_current, cv::FM_RANSAC,
        std::max(0.75, static_cast<double>(flow_threshold_) * 0.75),
        0.99, 2000, fundamental_inliers);
    if (fundamental.empty() || fundamental_inliers.total() != valid_previous.size()) {
        return mask;
    }
    const int fundamental_inlier_count = cv::countNonZero(fundamental_inliers);
    if (fundamental_inlier_count < static_cast<int>(0.6 * valid_previous.size())) {
        return mask;
    }
    last_residual_model_valid_ = true;

    std::vector<cv::Point2f> residuals(points.size(), cv::Point2f(0.0f, 0.0f));
    std::vector<uchar> residual_status(points.size(), 0);
    std::vector<cv::Point2f> mask_locations = points;
    for (std::size_t valid_index = 0; valid_index < valid_indices.size(); ++valid_index) {
        const int i = valid_indices[valid_index];
        const cv::Point2f& point = points[i];
        const cv::Point2f predicted(
            static_cast<float>(affine.at<double>(0, 0) * point.x +
                               affine.at<double>(0, 1) * point.y +
                               affine.at<double>(0, 2)),
            static_cast<float>(affine.at<double>(1, 0) * point.x +
                               affine.at<double>(1, 1) * point.y +
                               affine.at<double>(1, 2)));
        residuals[i] = last_tracked_[i] - predicted;
        const bool epipolar_outlier = fundamental_inliers.at<unsigned char>(
            static_cast<int>(valid_index), 0) == 0;
        residual_status[i] = epipolar_outlier ? 1 : 0;
        mask_locations[i] = last_tracked_[i];
    }
    mask = interpolateToMask(
        mask_locations, residuals, residual_status, curr_gray.size());
    last_static_residual_mask_ = mask.clone();
    return mask;
}

cv::Mat OpticalFlowLK::getFullFrameResidualMotionMask(
    const cv::Mat& prev_gray,
    const cv::Mat& curr_gray,
    const cv::Mat& static_model_roi,
    float high_confidence_scale)
{
    last_residual_model_valid_ = false;
    cv::Mat mask = cv::Mat::zeros(curr_gray.size(), CV_8UC1);
    last_static_residual_mask_ = mask.clone();
    last_high_confidence_residual_mask_ = mask.clone();
    if (prev_gray.empty() || curr_gray.empty() ||
        prev_gray.size() != curr_gray.size() ||
        static_model_roi.empty() ||
        static_model_roi.type() != CV_8UC1 ||
        static_model_roi.size() != prev_gray.size()) {
        return mask;
    }

    const std::vector<cv::Point2f> points =
        generateGridPoints(prev_gray.size(), grid_step_);
    if (points.size() < 6) {
        return mask;
    }

    compute(prev_gray, curr_gray, points);
    std::vector<cv::Point2f> backward_points;
    std::vector<uchar> backward_status;
    std::vector<float> backward_error;
    cv::calcOpticalFlowPyrLK(
        curr_gray, prev_gray, last_tracked_, backward_points,
        backward_status, backward_error, cv::Size(win_size_, win_size_),
        max_level_, term_criteria_);

    std::vector<int> valid_indices;
    std::vector<int> model_index_by_point(points.size(), -1);
    std::vector<cv::Point2f> model_previous;
    std::vector<cv::Point2f> model_current;
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (!last_status_[i] || !backward_status[i] ||
            cv::norm(points[i] - backward_points[i]) > 1.5f) {
            continue;
        }
        valid_indices.push_back(static_cast<int>(i));
        const int x = cvRound(points[i].x);
        const int y = cvRound(points[i].y);
        if (x >= 0 && x < static_model_roi.cols &&
            y >= 0 && y < static_model_roi.rows &&
            static_model_roi.at<uchar>(y, x) != 0) {
            model_index_by_point[i] =
                static_cast<int>(model_previous.size());
            model_previous.push_back(points[i]);
            model_current.push_back(last_tracked_[i]);
        }
    }
    if (model_previous.size() < 6) {
        return mask;
    }

    cv::Mat affine_inliers;
    const cv::Mat affine = cv::estimateAffinePartial2D(
        model_previous, model_current, affine_inliers, cv::RANSAC,
        std::max(1.0, static_cast<double>(flow_threshold_)),
        2000, 0.99, 10);
    if (affine.empty()) {
        return mask;
    }

    cv::Mat fundamental_inliers;
    const cv::Mat fundamental = cv::findFundamentalMat(
        model_previous, model_current, cv::FM_RANSAC,
        std::max(0.75, static_cast<double>(flow_threshold_) * 0.75),
        0.99, 2000, fundamental_inliers);
    if (fundamental.empty() ||
        fundamental_inliers.total() != model_previous.size() ||
        cv::countNonZero(fundamental_inliers) <
            static_cast<int>(0.6 * model_previous.size())) {
        return mask;
    }

    std::vector<cv::Point2f> residuals(
        points.size(), cv::Point2f(0.0f, 0.0f));
    std::vector<uchar> static_residual_status(points.size(), 0);
    std::vector<uchar> recovery_guard_status(points.size(), 0);
    std::vector<uchar> high_confidence_guard_status(points.size(), 0);
    std::vector<cv::Point2f> mask_locations = points;
    const float high_confidence_threshold =
        flow_threshold_ * std::max(1.0f, high_confidence_scale);
    for (const int i : valid_indices) {
        const cv::Point2f& point = points[i];
        const cv::Point2f predicted(
            static_cast<float>(
                affine.at<double>(0, 0) * point.x +
                affine.at<double>(0, 1) * point.y +
                affine.at<double>(0, 2)),
            static_cast<float>(
                affine.at<double>(1, 0) * point.x +
                affine.at<double>(1, 1) * point.y +
                affine.at<double>(1, 2)));
        residuals[i] = last_tracked_[i] - predicted;
        mask_locations[i] = last_tracked_[i];

        const int x = cvRound(point.x);
        const int y = cvRound(point.y);
        const bool trusted_for_model =
            x >= 0 && x < static_model_roi.cols &&
            y >= 0 && y < static_model_roi.rows &&
            static_model_roi.at<uchar>(y, x) != 0;
        bool residual_motion = false;
        if (trusted_for_model) {
            const int model_index = model_index_by_point[i];
            if (model_index >= 0 &&
                fundamental_inliers.at<unsigned char>(
                    model_index, 0) == 0) {
                static_residual_status[i] = 1;
                residual_motion = true;
            }
        } else if (cv::norm(residuals[i]) > flow_threshold_) {
            recovery_guard_status[i] = 1;
            residual_motion = true;
        }
        if (residual_motion &&
            cv::norm(residuals[i]) > high_confidence_threshold) {
            high_confidence_guard_status[i] = 1;
        }
    }

    last_residual_model_valid_ = true;
    last_static_residual_mask_ = interpolateToMask(
        mask_locations, residuals, static_residual_status,
        curr_gray.size());
    cv::Mat recovery_guard = interpolateToMask(
        mask_locations, residuals, recovery_guard_status,
        curr_gray.size());
    last_high_confidence_residual_mask_ = interpolateToMask(
        mask_locations, residuals, high_confidence_guard_status,
        curr_gray.size(), high_confidence_threshold);
    cv::bitwise_or(
        last_static_residual_mask_, recovery_guard, mask);
    return mask;
}

cv::Mat OpticalFlowLK::interpolateToMask(const std::vector<cv::Point2f>& prev_points,
                                          const std::vector<cv::Point2f>& flow_vectors,
                                          const std::vector<uchar>& status,
                                          const cv::Size& image_size,
                                          float threshold)
{
    cv::Mat mask = cv::Mat::zeros(image_size, CV_8UC1);

    if (prev_points.size() != flow_vectors.size() ||
        prev_points.size() != status.size()) {
        return mask;
    }

    // Create a flow magnitude map
    cv::Mat flow_mag = cv::Mat::zeros(image_size, CV_32FC1);

    // For each tracked point, compute flow magnitude and spread to neighbors
    for (size_t i = 0; i < prev_points.size(); ++i) {
        if (!status[i]) continue;

        float mag = std::sqrt(flow_vectors[i].x * flow_vectors[i].x +
                             flow_vectors[i].y * flow_vectors[i].y);

        int x = static_cast<int>(prev_points[i].x);
        int y = static_cast<int>(prev_points[i].y);

        // Spread to a small region around the point
        int half_step = grid_step_ / 2;
        for (int dy = -half_step; dy <= half_step; ++dy) {
            for (int dx = -half_step; dx <= half_step; ++dx) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < image_size.width &&
                    ny >= 0 && ny < image_size.height) {
                    // Use max to handle overlapping regions
                    flow_mag.at<float>(ny, nx) = std::max(flow_mag.at<float>(ny, nx), mag);
                }
            }
        }
    }

    // Optional: Apply Gaussian blur for smoother mask
    cv::GaussianBlur(flow_mag, flow_mag, cv::Size(5, 5), 0);

    const float active_threshold =
        threshold > 0.0f ? threshold : flow_threshold_;

    // Threshold to binary mask
    for (int y = 0; y < image_size.height; ++y) {
        const float* flow_row = flow_mag.ptr<float>(y);
        uchar* mask_row = mask.ptr<uchar>(y);

        for (int x = 0; x < image_size.width; ++x) {
            mask_row[x] = (flow_row[x] > active_threshold) ? 1 : 0;
        }
    }

    // Dilate to fill gaps
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::dilate(mask, mask, kernel);

    return mask;
}

std::vector<cv::Point2f> OpticalFlowLK::generateGridPoints(const cv::Size& image_size,
                                                            int step,
                                                            const cv::Mat& mask)
{
    std::vector<cv::Point2f> points;

    // Reserve approximate number of points
    int approx_count = (image_size.width / step) * (image_size.height / step);
    points.reserve(approx_count);

    for (int y = step / 2; y < image_size.height; y += step) {
        for (int x = step / 2; x < image_size.width; x += step) {
            // Check mask if provided
            if (!mask.empty()) {
                if (mask.at<uchar>(y, x) == 0) {
                    continue;
                }
            }
            points.emplace_back(static_cast<float>(x), static_cast<float>(y));
        }
    }

    return points;
}

cv::Mat OpticalFlowLK::visualizeFlow(const std::vector<cv::Point2f>& flow_vectors,
                                      const std::vector<cv::Point2f>& prev_points,
                                      const cv::Size& image_size) const
{
    cv::Mat vis = cv::Mat::zeros(image_size, CV_8UC3);

    if (flow_vectors.size() != prev_points.size()) {
        return vis;
    }

    for (size_t i = 0; i < prev_points.size(); ++i) {
        float mag = std::sqrt(flow_vectors[i].x * flow_vectors[i].x +
                             flow_vectors[i].y * flow_vectors[i].y);

        cv::Point2f end_pt = prev_points[i] + flow_vectors[i];

        // Color based on motion: green for static, red for motion
        cv::Scalar color;
        if (mag > flow_threshold_) {
            // Red for motion (intensity based on magnitude)
            int intensity = std::min(255, static_cast<int>(mag * 50));
            color = cv::Scalar(0, 0, intensity);
        } else {
            // Green for static
            color = cv::Scalar(0, 100, 0);
        }

        cv::line(vis, prev_points[i], end_pt, color, 1);
        cv::circle(vis, prev_points[i], 2, color, -1);
    }

    return vis;
}

} // namespace DyGeoFusion
