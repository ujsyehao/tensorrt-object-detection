#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>

#include <stdlib.h>
#include <time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

#include "pluginImplement.h"
#include "tensorNet.h"

class Detect {
  public:
    Detect(const char* model, const char* weight, int model_width, int model_height, float threshold);
    ~Detect();
    void ImgInference(cv::Mat &img);

  public:
    // tensorrt model
    const char* model_;
    const char* weight_;
    const char* INPUT_BLOB_NAME_;
    const char* OUTPUT_BLOB_NAME_;
    const int BATCH_SIZE_ = 1; //static uint32_t

    void* imgData_; // input cpu pointer
    void* imgCUDA_; // input gpu pointer
    float* output_; // output pointer

    // model input
    int width_;
    int height_;

    // detect threshold
    float threshold_;

    // input size
    size_t size_;

    // tensorrt network
    TensorNet* tensorNet_;

    // detection output results;
    //vector<vector<float> > results_;
};

class Timer {
  public:
    void tic() {
        start_ticking_ = true;
        start_ = std::chrono::high_resolution_clock::now();
    }
    void toc() {
        if (!start_ticking_)return;
        end_ = std::chrono::high_resolution_clock::now();
        start_ticking_ = false;
        t = std::chrono::duration<double, std::milli>(end_ - start_).count();
    }
    double t;
  private:
    bool start_ticking_ = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};
