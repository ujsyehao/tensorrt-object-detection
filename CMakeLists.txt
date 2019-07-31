CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
project(tensorrt-ssd)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")

set(CUDA_NVCC_FLAGS -arch=compute_61 -O3 -use_fast_math -Xcompiler  -std=c++11) #note: update later

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDA REQUIRED)

# set opencv
set(OPENCV_INCLUDE_DIR ./lib/opencv-3.4/include/opencv ./lib/opencv-3.4/include/opencv2)
set(OPENCV_LIBS_DIR ./lib/opencv-3.4/lib)
set(OPENCV_LIBS opencv_highgui opencv_imgcodecs opencv_imgproc opencv_core opencv_videoio)

# set tensorrt
set(TENSORRT_INCLUDE_DIR ./lib/tensorrt-5.0.2.6/include)
set(TENSORRT_LIBS_DIR ./lib/tensorrt-5.0.2.6/lib)
set(TENSORRT_LIBS nvinfer nvcaffe_parser nvinfer_plugin nvparsers)

# set cuda
set(CUDA_DIR /usr/local/cuda-9.0)

# release
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)
set(CMAKE_INSTALL_RPATH ./lib/opencv-3.4/lib ./lib/tensorrt-5.0.2.6/lib)
set(CMAKE_BUILD_WITH_INSTALL_RPATH true)

include_directories (
	${OPENCV_INCLUDE_DIR}
	${TENSORRT_INCLUDE_DIR}
	${CUDA_DIR}/include
	./include
	/usr/include/freetype2
)

link_directories(${OPENCV_LIBS_DIR} ${TENSORRT_LIBS_DIR} ${CUDA_DIR}/lib64)

FILE(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "./main.cpp")
FILE(GLOB_RECURSE CU "src/*.cu")

CUDA_COMPILE(CU_OBJS ${CU})
add_executable(tensorrt-ssd ${SOURCE_FILES} ${CU})

target_link_libraries(tensorrt-ssd ${CU_OBJS} ${OPENCV_LIBS} ${TENSORRT_LIBS} cublas cudnn cuda cudart glog freetype)
