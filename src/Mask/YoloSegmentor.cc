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

#include "include/Mask/YoloSegmentor.h"
#include "include/Mask/yolov8.h"
#include "include/Mask/common.hpp"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <future>
#include <stdexcept>
#include <thread>

namespace DyGeoFusion
{

// COCO 80 class names
const std::vector<std::string> YoloSegmentor::COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

// Dynamic category for each COCO class
const std::vector<DynamicCategory> YoloSegmentor::COCO_DYNAMIC_CATEGORIES = {
    DynamicCategory::HIGHLY_DYNAMIC,       // 0: person
    DynamicCategory::HIGHLY_DYNAMIC,       // 1: bicycle
    DynamicCategory::HIGHLY_DYNAMIC,       // 2: car
    DynamicCategory::HIGHLY_DYNAMIC,       // 3: motorcycle
    DynamicCategory::HIGHLY_DYNAMIC,       // 4: airplane
    DynamicCategory::HIGHLY_DYNAMIC,       // 5: bus
    DynamicCategory::HIGHLY_DYNAMIC,       // 6: train
    DynamicCategory::HIGHLY_DYNAMIC,       // 7: truck
    DynamicCategory::HIGHLY_DYNAMIC,       // 8: boat
    DynamicCategory::STATIC,               // 9: traffic light
    DynamicCategory::STATIC,               // 10: fire hydrant
    DynamicCategory::STATIC,               // 11: stop sign
    DynamicCategory::STATIC,               // 12: parking meter
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 13: bench
    DynamicCategory::HIGHLY_DYNAMIC,       // 14: bird
    DynamicCategory::HIGHLY_DYNAMIC,       // 15: cat
    DynamicCategory::HIGHLY_DYNAMIC,       // 16: dog
    DynamicCategory::HIGHLY_DYNAMIC,       // 17: horse
    DynamicCategory::HIGHLY_DYNAMIC,       // 18: sheep
    DynamicCategory::HIGHLY_DYNAMIC,       // 19: cow
    DynamicCategory::HIGHLY_DYNAMIC,       // 20: elephant
    DynamicCategory::HIGHLY_DYNAMIC,       // 21: bear
    DynamicCategory::HIGHLY_DYNAMIC,       // 22: zebra
    DynamicCategory::HIGHLY_DYNAMIC,       // 23: giraffe
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 24: backpack
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 25: umbrella
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 26: handbag
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 27: tie
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 28: suitcase
    DynamicCategory::HIGHLY_DYNAMIC,       // 29: frisbee
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 30: skis
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 31: snowboard
    DynamicCategory::HIGHLY_DYNAMIC,       // 32: sports ball
    DynamicCategory::HIGHLY_DYNAMIC,       // 33: kite
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 34: baseball bat
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 35: baseball glove
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 36: skateboard
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 37: surfboard
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 38: tennis racket
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 39: bottle
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 40: wine glass
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 41: cup
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 42: fork
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 43: knife
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 44: spoon
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 45: bowl
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 46: banana
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 47: apple
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 48: sandwich
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 49: orange
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 50: broccoli
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 51: carrot
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 52: hot dog
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 53: pizza
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 54: donut
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 55: cake
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 56: chair
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 57: couch
    DynamicCategory::STATIC,               // 58: potted plant
    DynamicCategory::STATIC,               // 59: bed
    DynamicCategory::STATIC,               // 60: dining table
    DynamicCategory::STATIC,               // 61: toilet
    DynamicCategory::STATIC,               // 62: tv
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 63: laptop
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 64: mouse
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 65: remote
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 66: keyboard
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 67: cell phone
    DynamicCategory::STATIC,               // 68: microwave
    DynamicCategory::STATIC,               // 69: oven
    DynamicCategory::STATIC,               // 70: toaster
    DynamicCategory::STATIC,               // 71: sink
    DynamicCategory::STATIC,               // 72: refrigerator
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 73: book
    DynamicCategory::STATIC,               // 74: clock
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 75: vase
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 76: scissors
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 77: teddy bear
    DynamicCategory::POTENTIALLY_DYNAMIC,  // 78: hair drier
    DynamicCategory::POTENTIALLY_DYNAMIC   // 79: toothbrush
};

// Dynamic class IDs (classes that are typically dynamic)
const std::vector<int> YoloSegmentor::DYNAMIC_CLASS_IDS = {
    0,   // person
    1,   // bicycle
    2,   // car
    3,   // motorcycle
    5,   // bus
    7,   // truck
    14,  // bird
    15,  // cat
    16,  // dog
    17,  // horse
    18,  // sheep
    19   // cow
};

YoloSegmentor::YoloSegmentor(const std::string& model_path,
                             float conf_threshold,
                             float nms_threshold)
    : yolo_trt_(nullptr),
      is_ready_(false),
      conf_threshold_(conf_threshold),
      nms_threshold_(nms_threshold),
      last_inference_time_ms_(0.0f)
{
    if (model_path.empty()) {
        std::cerr << "[YoloSegmentor] No model path provided" << std::endl;
        return;
    }

    // Check if file exists
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "[YoloSegmentor] Model file not found: " << model_path << std::endl;
        return;
    }

    // Check if it's a TensorRT engine file
    if (model_path.find(".engine") == std::string::npos) {
        std::cerr << "[YoloSegmentor] Expected .engine file for TensorRT, got: " << model_path << std::endl;
        return;
    }

    try {
        std::cout << "[YoloSegmentor] Loading TensorRT engine: " << model_path << std::endl;
        yolo_trt_ = std::make_shared<YOLOv8>(model_path);

        // Warmup with timeout mechanism
        std::cout << "[YoloSegmentor] Starting warmup inference (timeout: 10s)..." << std::endl;

        std::future<void> warmup_future = std::async(std::launch::async, [this]() {
            try {
                yolo_trt_->make_pipe(true);
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Warmup failed: ") + e.what());
            }
        });

        if (warmup_future.wait_for(std::chrono::seconds(10)) == std::future_status::timeout) {
            std::cerr << "[YoloSegmentor] Warmup timeout after 10 seconds - TensorRT may be incompatible" << std::endl;
            std::cerr << "[YoloSegmentor] Please regenerate .engine file with current CUDA/TensorRT version" << std::endl;
            yolo_trt_ = nullptr;
            is_ready_ = false;
            return;
        }

        warmup_future.get();
        is_ready_ = true;
        std::cout << "[YoloSegmentor] TensorRT engine loaded successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[YoloSegmentor] Failed to load TensorRT engine: " << e.what() << std::endl;
        yolo_trt_ = nullptr;
        is_ready_ = false;
    }
}

YoloSegmentor::~YoloSegmentor()
{
    // Cleanup handled by shared_ptr
}

std::vector<DetectionResult> YoloSegmentor::infer(const cv::Mat& image)
{
    std::vector<DetectionResult> results;

    if (!is_ready_ || !yolo_trt_) {
        throw std::runtime_error(
            "YOLO inference requested before the TensorRT engine was ready");
    }

    auto start_time = std::chrono::steady_clock::now();

    try {
        // Copy image to TensorRT
        yolo_trt_->copy_from_Mat(image);

        // Run inference
        yolo_trt_->infer();

        // Get detections
        std::vector<Object> objs;
        yolo_trt_->postprocess(objs);

        // Debug: print detection count
        static int debug_count = 0;
        if (debug_count < 5 || debug_count % 100 == 0) {
            std::cout << "[YoloSegmentor] Frame " << debug_count
                      << ": detected " << objs.size() << " objects" << std::endl;
            for (const auto& obj : objs) {
                bool is_dyn = false;
                for (int cid : DYNAMIC_CLASS_IDS) {
                    if (obj.label == cid) { is_dyn = true; break; }
                }
                std::cout << "  class=" << obj.label << " (" << getClassName(obj.label) << ")"
                          << " prob=" << obj.prob
                          << " dynamic=" << (is_dyn ? "YES" : "no") << std::endl;
            }
        }
        debug_count++;

        // Convert to DetectionResult
        for (const auto& obj : objs) {
            // Check if this is a dynamic class
            bool is_dynamic = false;
            for (int class_id : DYNAMIC_CLASS_IDS) {
                if (obj.label == class_id) {
                    is_dynamic = true;
                    break;
                }
            }

            // Filter by confidence and dynamic class
            if (is_dynamic && obj.prob >= conf_threshold_) {
                DetectionResult det;
                det.class_id = obj.label;
                det.confidence = obj.prob;
                det.bbox = cv::Rect(
                    static_cast<int>(obj.rect.x),
                    static_cast<int>(obj.rect.y),
                    static_cast<int>(obj.rect.width),
                    static_cast<int>(obj.rect.height)
                );
                det.class_name = getClassName(obj.label);
                results.push_back(det);
            }
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("YOLO TensorRT inference failed: ") + e.what());
    }

    auto end_time = std::chrono::steady_clock::now();
    last_inference_time_ms_ = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count() / 1000.0f;

    return results;
}

cv::Mat YoloSegmentor::getDynamicMask(const cv::Mat& image,
                                       bool include_potentially_dynamic)
{
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);

    auto detections = infer(image);

    for (const auto& det : detections) {
        DynamicCategory cat = getDynamicCategory(det.class_id);

        bool is_dynamic = (cat == DynamicCategory::HIGHLY_DYNAMIC) ||
                         (include_potentially_dynamic && cat == DynamicCategory::POTENTIALLY_DYNAMIC);

        if (is_dynamic) {
            // Clamp bbox to image bounds
            cv::Rect safe_bbox = det.bbox & cv::Rect(0, 0, image.cols, image.rows);
            if (safe_bbox.area() > 0) {
                mask(safe_bbox).setTo(1);
            }
        }
    }

    // Dilate to cover edges
    if (cv::countNonZero(mask) > 0) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::dilate(mask, mask, kernel);
    }

    return mask;
}

DynamicCategory YoloSegmentor::getDynamicCategory(int class_id)
{
    if (class_id >= 0 && class_id < static_cast<int>(COCO_DYNAMIC_CATEGORIES.size())) {
        return COCO_DYNAMIC_CATEGORIES[class_id];
    }
    return DynamicCategory::STATIC;
}

std::string YoloSegmentor::getClassName(int class_id)
{
    if (class_id >= 0 && class_id < static_cast<int>(COCO_CLASSES.size())) {
        return COCO_CLASSES[class_id];
    }
    return "unknown";
}

} // namespace DyGeoFusion
