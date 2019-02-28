CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
project(tensorrt-ssd)

#set(CMAKE_CXX_COMPILER "g++")
#add_definitions(-std=c++11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -msse2 -msse3 -msse4 -fvisibility=hidden -fopenmp -O3 -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdata-sections -ffunction-sections -fvisibility=hidden -fPIC")
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")

set(CUDA_NVCC_FLAGS -arch=compute_30 -O3 -use_fast_math -Xcompiler -fPIC -std=c++11) #note: update later

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDA REQUIRED)

# set opencv
set(OPENCV_INCLUDE_DIR /usr/local/include/opencv /usr/local/include/opencv2)
set(OPENCV_LIBS_DIR /usr/local/lib)
set(OPENCV_LIBS opencv_highgui opencv_imgcodecs opencv_imgproc opencv_core)

# set tensorrt
set(TENSORRT_INCLUDE_DIR /opt/TensorRT-4.0.0.3/include)
set(TENSORRT_LIBS_DIR /opt/TensorRT-4.0.0.3/lib)
set(TENSORRT_LIBS nvinfer nvcaffe_parser nvinfer_plugin nvparsers)

# set cuda
set(CUDA_DIR /usr/local/cuda-9.0)

include_directories (
	${OPENCV_INCLUDE_DIR}
	${TENSORRT_INCLUDE_DIR}
	${CUDA_DIR}/include
	./include
)

link_directories(${OPENCV_LIBS_DIR} ${TENSORRT_LIBS_DIR} ${CUDA_DIR}/lib64)

FILE(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "./main.cpp")
FILE(GLOB_RECURSE CU "src/*.cu")

#cuda_add_executable(tensorrt-ssd ${SOURCE_FILES} ${CU} )
CUDA_COMPILE(CU_OBJS ${CU})
add_executable(tensorrt-ssd ${SOURCE_FILES} ${CU})

target_link_libraries(tensorrt-ssd ${CU_OBJS} ${OPENCV_LIBS} ${TENSORRT_LIBS} cublas cudnn cuda cudart glog)
