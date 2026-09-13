cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(dygeofusion-slam LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    # set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

set(CAFFE2_USE_CUDNN on)

# Uncomment the following line if you want to solve CMake Warning "Failed to compute shorthash for libnvrtc.so"
# set(PYTHON_EXECUTABLE "/usr/bin/python3")
set(
    DYNAGS_LIBTORCH_ROOT
    ""
    CACHE PATH
    "Path to an external LibTorch installation (for example 2.0.1+cu118)")
if(NOT DYNAGS_LIBTORCH_ROOT)
    message(FATAL_ERROR
        "Set -DDYNAGS_LIBTORCH_ROOT=/path/to/libtorch; the binary LibTorch package is not vendored.")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${DYNAGS_LIBTORCH_ROOT}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

# set(ENV{CUDA_LAUNCH_BLOCKING} 1)
# set(ENV{TORCH_USE_CUDA_DSA} 1)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
set(DYNAGS_OPENCV_DIR "" CACHE PATH "Optional OpenCV CMake package directory")
if(DYNAGS_OPENCV_DIR)
    set(OpenCV_DIR "${DYNAGS_OPENCV_DIR}")
endif()
find_package(OpenCV 4.7.0 EXACT)
find_package(jsoncpp REQUIRED)

find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)

option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    enable_testing()
endif()

find_package(realsense2)

set(ORB_SLAM3_SOURCE_DIR "${PROJECT_SOURCE_DIR}/ORB-SLAM3")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${ORB_SLAM3_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}/include
    ${ORB_SLAM3_SOURCE_DIR}/include/CameraModels
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR})

##################################################################################
##  Build the gaussian splatting component libraries to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(simple_knn SHARED
    third_party/simple-knn/simple_knn.cu
    third_party/simple-knn/simple_knn.h
    third_party/simple-knn/spatial.cu
    third_party/simple-knn/spatial.h)
# target_compile_features(simple_knn PUBLIC cxx_std_17)
target_link_libraries(simple_knn "${TORCH_LIBRARIES}")

add_library(cuda_rasterizer SHARED
    include/operate_points.h
    src/operate_points.cu
    include/rasterize_points.h
    src/rasterize_points.cu
    include/stereo_vision.h
    src/stereo_vision.cu
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/backward.cu
    cuda_rasterizer/backward.h
    cuda_rasterizer/config.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/operate_points.h
    cuda_rasterizer/rasterizer.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/stereo_vision.h)
set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "75;86")
# target_compile_features(cuda_rasterizer PUBLIC cxx_std_17)
target_include_directories(cuda_rasterizer PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda_rasterizer
    glm::glm
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen)

##################################################################################
##  Build the ImGui library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(imgui SHARED
    viewer/imgui/imconfig.h
    viewer/imgui/imgui_demo.cpp
    viewer/imgui/imgui_draw.cpp
    viewer/imgui/imgui_impl_glfw.cpp
    viewer/imgui/imgui_impl_glfw.h
    viewer/imgui/imgui_impl_opengl3_loader.h
    viewer/imgui/imgui_impl_opengl3.cpp
    viewer/imgui/imgui_impl_opengl3.h
    viewer/imgui/imgui_internal.h
    viewer/imgui/imgui_tables.cpp
    viewer/imgui/imgui_widgets.cpp
    viewer/imgui/imgui.cpp
    viewer/imgui/imgui.h
    viewer/imgui/imstb_rectpack.h
    viewer/imgui/imstb_textedit.h
    viewer/imgui/imstb_truetype.h)
target_link_libraries(imgui glfw OpenGL::GL)

##################################################################################
##  Build the DynamicMaskRefiner library (NEW)
##################################################################################

# TensorRT for YOLOv8
set(TENSORRT_DIR "/usr/local/TensorRT-8.6.1.6")
find_library(NVINFER_LIB nvinfer HINTS ${TENSORRT_DIR}/lib)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TENSORRT_DIR}/lib)
find_library(NVPARSERS_LIB nvparsers HINTS ${TENSORRT_DIR}/lib)
find_library(NVONNXPARSER_LIB nvonnxparser HINTS ${TENSORRT_DIR}/lib)

add_library(dynamic_mask_refiner SHARED
    include/Mask/MaskConfig.h
    include/Mask/DynamicMaskRefiner.h
    include/Mask/YoloSegmentor.h
    include/Mask/OpticalFlowLK.h
    include/Mask/common.hpp
    include/Mask/yolov8.h
    include/Mask/filesystem.hpp
    src/Mask/MaskConfig.cc
    src/Mask/DynamicMaskRefiner.cc
    src/Mask/YoloSegmentor.cc
    src/Mask/OpticalFlowLK.cc
    src/Mask/yolov8.cc)
target_include_directories(dynamic_mask_refiner PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
    ${TENSORRT_DIR}/include
    ${CUDA_INCLUDE_DIRS})
target_link_libraries(dynamic_mask_refiner
    ${OpenCV_LIBRARIES}
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen
    ${NVINFER_LIB}
    ${NVINFER_PLUGIN_LIB}
    ${CUDA_LIBRARIES})

##################################################################################
##  Build the Motion3D library (NEW - 3D Gaussian Motion Segmentation)
##################################################################################

add_library(motion3d SHARED
    include/Motion3D/BackgroundSupportRouter.h
    include/Motion3D/MotionMarginalizedPosePrior.h
    src/Motion3D/BackgroundSupportRouter.cc
    src/Motion3D/MotionMarginalizedPosePrior.cc)
target_include_directories(motion3d PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus)
target_link_libraries(motion3d
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen)

# Legacy diagnostic only: Gaussian parameter drift is not object motion and is
# intentionally kept out of the default runtime path.
add_library(gaussian_motion_tracker SHARED
    include/Motion3D/GaussianMotionTracker.h
    src/Motion3D/GaussianMotionTracker.cc)
target_include_directories(gaussian_motion_tracker PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus)
target_link_libraries(gaussian_motion_tracker
    ${OpenCV_LIBRARIES}
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen)

##################################################################################
##  Build the gaussian mapper library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(gaussian_mapper SHARED
    third_party/tinyply/tinyply.h
    third_party/tinyply/tinyply.cpp
    include/gaussian_keyframe.h
    include/gaussian_model.h
    include/gaussian_parameters.h
    include/gaussian_rasterizer.h
    include/gaussian_renderer.h
    include/gaussian_scene.h
    include/gaussian_trainer.h
    include/gaussian_mapper.h
    include/general_utils.h
    include/graphics_utils.h
    include/loss_utils.h
    include/sh_utils.h
    include/tensor_utils.h
    include/camera.h
    include/point_cloud.h
    include/point2d.h
    include/point3d.h
    include/types.h
    src/gaussian_keyframe.cpp
    src/gaussian_model.cpp
    src/gaussian_parameters.cpp
    src/gaussian_rasterizer.cpp
    src/gaussian_renderer.cpp
    src/gaussian_scene.cpp
    src/gaussian_trainer.cpp
    src/gaussian_mapper.cpp)
target_link_libraries(gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen
    simple_knn
    cuda_rasterizer
    dynamic_mask_refiner)

##################################################################################
##  Build the viewer library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(gaussian_viewer SHARED
    viewer/drawer_utils.h
    viewer/imgui_viewer.cpp
    viewer/imgui_viewer.h
    viewer/map_drawer.cpp
    viewer/map_drawer.h)
target_link_libraries(gaussian_viewer
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    gaussian_mapper
    imgui
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    glm::glm
    glfw
    OpenGL::GL)

##################################################################################
##  Build the test examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

# This is a C++ libtorch implementation of gaussian-splatting (https://github.com/graphdeco-inria/gaussian-splatting)
add_executable(train_colmap examples/train_colmap.cpp)
target_link_libraries(train_colmap
    gaussian_viewer
    gaussian_mapper)

add_executable(view_result examples/view_result.cpp)
target_link_libraries(view_result
    gaussian_viewer
    gaussian_mapper)

##################################################################################
##  Build the mapping examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

# Replica Monocular
add_executable(replica_mono examples/replica_mono.cpp)
target_link_libraries(replica_mono
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# Replica Monocular
add_executable(replica_rgbd examples/replica_rgbd.cpp)
target_link_libraries(replica_rgbd
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# TUM Monocular
add_executable(tum_mono examples/tum_mono.cpp)
target_link_libraries(tum_mono
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# TUM RGBD
add_executable(tum_rgbd examples/tum_rgbd.cpp)
target_link_libraries(tum_rgbd
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# EuRoC Stereo
add_executable(euroc_stereo examples/euroc_stereo.cpp)
target_link_libraries(euroc_stereo
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

##################################################################################
##  Build TUM RGBD Dynamic example (NEW - DyGeoFusion-SLAM+)
##################################################################################

add_executable(tum_rgbd_dynamic examples/tum_rgbd_dynamic.cpp)
target_link_libraries(tum_rgbd_dynamic
    gaussian_viewer
    gaussian_mapper
    dynamic_mask_refiner
    motion3d
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

add_executable(frozen_same_state_replay
    tools/frozen_same_state_replay.cpp)
target_link_libraries(frozen_same_state_replay
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen)

add_executable(checkpointed_fork_replay
    tools/checkpointed_fork_replay.cpp)
target_link_libraries(checkpointed_fork_replay
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen)

##################################################################################
##  Build Intel Realsense example
##################################################################################

# Intel Realsense
if(realsense2_FOUND)
    add_executable(realsense_rgbd examples/realsense_rgbd.cpp)
    target_include_directories(realsense_rgbd PUBLIC
        ${realsense_INCLUDE_DIR})
    target_link_libraries(realsense_rgbd
        gaussian_viewer
        gaussian_mapper
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
        ${realsense2_LIBRARY})
endif()

##################################################################################
##  Build Unit Tests (optional)
##################################################################################

if(BUILD_TESTS)
    add_executable(test_gaussian_motion_tracker
        tests/test_gaussian_motion_tracker.cpp)

    target_include_directories(test_gaussian_motion_tracker PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
        ${GTEST_INCLUDE_DIRS})

    target_link_libraries(test_gaussian_motion_tracker
        gaussian_motion_tracker
        gaussian_mapper
        GTest::GTest
        GTest::Main
        "${TORCH_LIBRARIES}"
        Eigen3::Eigen
        ${OpenCV_LIBRARIES})

    add_test(NAME GaussianMotionTrackerTest COMMAND test_gaussian_motion_tracker)

    add_executable(test_motion_marginalized_pose_prior
        tests/test_motion_marginalized_pose_prior.cpp)
    target_include_directories(test_motion_marginalized_pose_prior PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
        ${GTEST_INCLUDE_DIRS})
    target_link_libraries(test_motion_marginalized_pose_prior
        motion3d
        GTest::GTest
        GTest::Main
        Eigen3::Eigen
        ${OpenCV_LIBRARIES})
    add_test(NAME MotionMarginalizedPosePriorTest
        COMMAND test_motion_marginalized_pose_prior)

    add_executable(test_background_support_router
        tests/test_background_support_router.cpp)
    target_include_directories(test_background_support_router PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${GTEST_INCLUDE_DIRS})
    target_link_libraries(test_background_support_router
        motion3d
        GTest::GTest
        GTest::Main
        ${OpenCV_LIBRARIES})
    add_test(NAME BackgroundSupportRouterTest
        COMMAND test_background_support_router)

    add_executable(test_direct_rgbd_pose_validator
        tests/test_direct_rgbd_pose_validator.cpp)
    target_include_directories(test_direct_rgbd_pose_validator PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
        ${GTEST_INCLUDE_DIRS})
    target_link_libraries(test_direct_rgbd_pose_validator
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
        GTest::GTest
        GTest::Main
        Eigen3::Eigen
        ${OpenCV_LIBRARIES})
    add_test(NAME DirectRgbdPoseValidatorTest
        COMMAND test_direct_rgbd_pose_validator)

    add_executable(test_frozen_replay_packet
        tests/test_frozen_replay_packet.cpp)
    target_include_directories(test_frozen_replay_packet PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${ORB_SLAM3_SOURCE_DIR}
        ${ORB_SLAM3_SOURCE_DIR}/include
        ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
        ${GTEST_INCLUDE_DIRS})
    target_link_libraries(test_frozen_replay_packet
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
        ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        GTest::GTest
        GTest::Main
        Eigen3::Eigen
        ${OpenCV_LIBRARIES})
    add_test(NAME FrozenReplayPacketTest
        COMMAND test_frozen_replay_packet)

    add_executable(test_orb_pose_prior
        tests/test_orb_pose_prior.cpp)
    target_include_directories(test_orb_pose_prior PRIVATE
        ${ORB_SLAM3_SOURCE_DIR}
        ${ORB_SLAM3_SOURCE_DIR}/include
        ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
        ${GTEST_INCLUDE_DIRS})
    target_link_libraries(test_orb_pose_prior
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
        ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        GTest::GTest
        GTest::Main
        Eigen3::Eigen
        ${OpenCV_LIBRARIES})
    add_test(NAME OrbPosePriorTest COMMAND test_orb_pose_prior)

    add_executable(test_matched_sham_oracle
        tests/test_matched_sham_oracle.cpp)
    target_include_directories(test_matched_sham_oracle PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
        ${GTEST_INCLUDE_DIRS})
    target_link_libraries(test_matched_sham_oracle
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
        GTest::GTest
        GTest::Main
        Eigen3::Eigen)
    add_test(NAME MatchedShamOracleTest COMMAND test_matched_sham_oracle)

    add_executable(test_local_mapping_idle
        tests/test_local_mapping_idle.cpp)
    target_include_directories(test_local_mapping_idle PRIVATE
        ${ORB_SLAM3_SOURCE_DIR}
        ${ORB_SLAM3_SOURCE_DIR}/include
        ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
        ${GTEST_INCLUDE_DIRS})
    target_link_libraries(test_local_mapping_idle
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
        ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        GTest::GTest
        GTest::Main
        Eigen3::Eigen
        ${OpenCV_LIBRARIES})
    add_test(NAME LocalMappingIdleTest COMMAND test_local_mapping_idle)

    add_executable(test_optical_flow_lk
        tests/test_optical_flow_lk.cpp
        src/Mask/OpticalFlowLK.cc)
    target_include_directories(test_optical_flow_lk PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${GTEST_INCLUDE_DIRS})
    target_link_libraries(test_optical_flow_lk
        GTest::GTest
        GTest::Main
        ${OpenCV_LIBRARIES})
    add_test(NAME OpticalFlowLKTest COMMAND test_optical_flow_lk)

    add_executable(test_dynamic_mask_refiner
        tests/test_dynamic_mask_refiner.cpp)
    target_include_directories(test_dynamic_mask_refiner PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
        ${GTEST_INCLUDE_DIRS})
    target_link_libraries(test_dynamic_mask_refiner
        dynamic_mask_refiner
        GTest::GTest
        GTest::Main
        "${TORCH_LIBRARIES}"
        ${OpenCV_LIBRARIES})
    add_test(NAME DynamicMaskRefinerTest COMMAND test_dynamic_mask_refiner)

    add_executable(test_orb_mask_adaptive
        tests/test_orb_mask_adaptive.cpp)
    target_include_directories(test_orb_mask_adaptive PRIVATE
        ${ORB_SLAM3_SOURCE_DIR}/include
        ${GTEST_INCLUDE_DIRS})
    target_link_libraries(test_orb_mask_adaptive
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
        GTest::GTest
        GTest::Main
        ${OpenCV_LIBRARIES})
    add_test(NAME OrbMaskAdaptiveTest COMMAND test_orb_mask_adaptive)

    add_test(
        NAME ReproducibleBenchmarkAggregationTest
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_run_reproducible_benchmark.py)

    add_test(
        NAME RecoveredSupportOverlapCertificateTest
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_certify_recovered_support_overlap.py)

    add_test(
        NAME MappingHeldoutPreparationTest
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_prepare_mapping_heldout.py)

    add_test(
        NAME HeldoutRenderingMetricsTest
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_evaluate_heldout_rendering.py)

    add_test(
        NAME OccludedBackgroundProxyTest
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_evaluate_occluded_background_proxy.py)

    add_test(
        NAME Dyn19MapIntegritySummaryTest
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_summarize_dyn19_map_integrity.py)

    add_test(
        NAME MappingAblationSummaryTest
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_summarize_mapping_ablation.py)

    add_test(
        NAME MotionAblationEvidenceExportTest
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_export_motion_ablation_evidence.py)

    add_test(
        NAME BonnTrackingOfflineRecheckTest
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_recheck_bonn_tracking.py)

    add_test(
        NAME MotionPriorCounterfactualTest
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_evaluate_motion_prior_counterfactual.py)

    add_test(
        NAME SplaTAMCheckpointManifestTest
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_render_splatam_checkpoint.py)

    add_test(
        NAME NullInterventionEquivalenceTest
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_verify_null_intervention_equivalence.py)

    add_test(
        NAME CheckpointedForkEvaluationTest
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_evaluate_checkpointed_forks.py)

    add_test(
        NAME RecoveryEpisodeV2Test
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_recovery_episode_v2.py)

endif()
