#!/usr/bin/env bash

set -euo pipefail

JOBS="${JOBS:-$(nproc)}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
OPENCV_DIR="${DYNAGS_OPENCV_DIR:-}"
LIBTORCH_ROOT="${DYNAGS_LIBTORCH_ROOT:-}"

if [[ -z "$LIBTORCH_ROOT" ]]; then
    echo "Set DYNAGS_LIBTORCH_ROOT to an external LibTorch installation." >&2
    exit 2
fi

cmake -S ORB-SLAM3/Thirdparty/DBoW2 -B ORB-SLAM3/Thirdparty/DBoW2/build \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    ${OPENCV_DIR:+-DDYNAGS_OPENCV_DIR="$OPENCV_DIR"}
cmake --build ORB-SLAM3/Thirdparty/DBoW2/build -j"$JOBS"

cmake -S ORB-SLAM3 -B ORB-SLAM3/build \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    ${OPENCV_DIR:+-DDYNAGS_OPENCV_DIR="$OPENCV_DIR"}
cmake --build ORB-SLAM3/build -j"$JOBS"

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_TESTS=ON \
    -DDYNAGS_LIBTORCH_ROOT="$LIBTORCH_ROOT" \
    ${OPENCV_DIR:+-DDYNAGS_OPENCV_DIR="$OPENCV_DIR"}
cmake --build build -j"$JOBS"

echo "Build completed. Runtime vocabulary and TensorRT engine are external assets."
