cmake_minimum_required(VERSION 3.8)
project(test_libraries)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.2")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")

include_directories(
  ${CUDA_TOOLKIT_ROOT_DIR}/include  )

# Manually specify TensorRT paths (Jetson-specific)
set(TENSORRT_INCLUDE_DIRS "/usr/include/aarch64-linux-gnu")
set(TENSORRT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
include_directories(${TENSORRT_INCLUDE_DIRS})
link_directories(${TENSORRT_LIB_DIR})

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

include_directories(includes)

# add_executable(${PROJECT_NAME}
#   lib_test.cpp
# )

add_executable(${PROJECT_NAME}
  srcs/main.cpp
  srcs/utils.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  nvinfer
  nvinfer_plugin
  ${CUDA_LIBRARIES}
  stdc++fs
)
