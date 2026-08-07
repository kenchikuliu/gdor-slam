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

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Forward declaration for TensorRT YOLOv8
class YOLOv8;

namespace DyGeoFusion
{

/**
 * @brief Detection result from YOLO segmentation
 */
struct DetectionResult
{
    int class_id;           ///< Class ID
    float confidence;       ///< Detection confidence
    cv::Rect bbox;          ///< Bounding box
    cv::Mat mask;           ///< Instance segmentation mask (if available)
    std::string class_name; ///< Class name
};

/**
 * @brief Dynamic object class categories
 */
enum class DynamicCategory
{
    STATIC = 0,           ///< Always static (walls, floor, etc.)
    POTENTIALLY_DYNAMIC,  ///< Can move but often static (chairs, bags)
    HIGHLY_DYNAMIC        ///< Usually moving (people, animals, vehicles)
};

/**
 * @brief YoloSegmentor - YOLO-based semantic segmentation for dynamic object detection
 *
 * Uses TensorRT for GPU-accelerated inference.
 * Classifies detected objects into dynamic categories based on COCO class definitions.
 */
class YoloSegmentor
{
public:
    /**
     * @brief Construct YoloSegmentor
     * @param model_path Path to YOLO TensorRT engine file (.engine)
     * @param conf_threshold Confidence threshold
     * @param nms_threshold NMS threshold
     */
    YoloSegmentor(const std::string& model_path,
                  float conf_threshold = 0.5f,
                  float nms_threshold = 0.45f);

    /**
     * @brief Destructor
     */
    ~YoloSegmentor();

    /**
     * @brief Run inference on input image
     * @param image Input BGR image
     * @return Vector of detection results
     */
    std::vector<DetectionResult> infer(const cv::Mat& image);

    /**
     * @brief Generate binary dynamic mask from detections
     * @param image Input image (for size reference)
     * @param include_potentially_dynamic Include potentially dynamic objects
     * @return cv::Mat Binary mask (CV_8UC1, 1=dynamic candidate, 0=static)
     */
    cv::Mat getDynamicMask(const cv::Mat& image,
                           bool include_potentially_dynamic = false);

    /**
     * @brief Get dynamic category for a class ID
     * @param class_id COCO class ID
     * @return DynamicCategory
     */
    static DynamicCategory getDynamicCategory(int class_id);

    /**
     * @brief Get class name for a class ID
     * @param class_id COCO class ID
     * @return Class name string
     */
    static std::string getClassName(int class_id);

    /**
     * @brief Check if model is loaded successfully
     */
    bool isReady() const { return is_ready_; }

    /**
     * @brief Get last inference time in milliseconds
     */
    float getLastInferenceTime() const { return last_inference_time_ms_; }

private:
    std::shared_ptr<YOLOv8> yolo_trt_;
    bool is_ready_;
    float conf_threshold_;
    float nms_threshold_;
    float last_inference_time_ms_;

    // COCO class names (80 classes)
    static const std::vector<std::string> COCO_CLASSES;

    // Dynamic category mapping for COCO classes
    static const std::vector<DynamicCategory> COCO_DYNAMIC_CATEGORIES;

    // Dynamic class IDs (subset of COCO that are typically dynamic)
    static const std::vector<int> DYNAMIC_CLASS_IDS;
};

} // namespace DyGeoFusion
