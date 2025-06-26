#!/bin/bash

set -e
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake ..
cmake --build .
echo "Build completed successfully."
