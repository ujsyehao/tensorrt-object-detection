#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

#include "pluginImplement.h"
#include "tensorNet.h"


//const char* model = "./model/1.tensorcache";
const char* model = "./model/deploy.prototxt";
const char* weight = "./model/VOC0712_refinedet_vgg16_320x320_final.caffemodel";


const char* INPUT_BLOB_NAME = "data";
//const char* OUTPUT_BLOB_NAME = "arm_conf_flatten";
const char* OUTPUT_BLOB_NAME = "detection_out";

static const uint32_t BATCH_SIZE = 1;

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
        //std::cout << "Time: " << t << " ms" << std::endl;
    }
    double t;
private:
    bool start_ticking_ = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};


/* *
 * @TODO: unifiedMemory is used here under -> ( cudaMallocManaged )
 * */
float* allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    //std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged(&ptr, size*sizeof(float)));
    return ptr;
}


void loadImg(cv::Mat &input, int re_width, int re_height, float *data_unifrom, const float3 mean, const float scale )
{
    int i;
    int j;
    int line_offset;
    int offset_g;
    int offset_r;
    cv::Mat dst;

    unsigned char *line = NULL;
    float *unifrom_data = data_unifrom;

    cv::resize( input, dst, cv::Size( re_width, re_height ), (0.0), (0.0), cv::INTER_LINEAR );
    offset_g = re_width * re_height;
    offset_r = re_width * re_height * 2;
	//std::cout << "offset_g: " << offset_g << std::endl;
    //std::cout << "offset_r: " << offset_r << std::endl;
    for ( i = 0; i < re_height; ++i )
    {
        line = dst.ptr<uchar>(i);
        line_offset = i * re_width;
        for( j = 0; j < re_width; ++j )
        {
            // b
            unifrom_data[ line_offset + j  ] =  (float(line[ j * 3 ]) - mean.x) ;
            // g
            unifrom_data[ offset_g + line_offset + j ] = (float(line[ j * 3 + 1 ] - mean.y)) ;
            // r
            unifrom_data[ offset_r + line_offset + j ] = (float(line[ j * 3 + 2 ]) - mean.z) ;
        }
    }
    //for (int i = 0; i < 10; i++)
    //    std::cout << unifrom_data[i]  << std::endl;
    
}


int main()
{
    std::vector<std::string> output_vector = {OUTPUT_BLOB_NAME};
    
    TensorNet tensorNet;
    // create network
    tensorNet.LoadNetwork(model, weight, INPUT_BLOB_NAME, output_vector, BATCH_SIZE);    

    // load network
    //tensorNet.LoadNetwork(model, INPUT_BLOB_NAME, output_vector);

    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsOut  = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);
    DimsCHW dimsOut1 = DimsCHW(1, 500, 7);

    //float* data    = allocateMemory(dimsData, (char*)"input blob");
    //float* output  = allocateMemory(dimsOut, (char*)"output blob");
    //float* output = NULL;
    float* output = allocateMemory(dimsOut1, (char*)"fake output blob");
    std::cout << "origin output pointer: " << output << std::endl;

    std::cout << "input dimension: " << dimsData.c() << " " << dimsData.h() << " " << dimsData.w() << std::endl;
    std::cout << "output dimension: " << dimsOut.c() << " " << dimsOut.h() << " " << dimsOut.w() << " fuck you" << std::endl;

    int height = 320;
    int width  = 320;

    cv::Mat frame, src;
    void* imgCUDA;

    Timer timer;

    std::string imgFile = "./test.jpg";
    
    frame = cv::imread(imgFile);
    //std::cout << frame.type() << std::endl;

    cv::Mat srcImg = frame.clone();

    const size_t size = width * height * 3 * sizeof(float);

    cudaMalloc(&imgCUDA, size);

    void* imgData = malloc(size);
    memset(imgData, 0, size);

    loadImg(frame, height, width, (float*)imgData, make_float3(104, 117, 123), 0.007843);
    
    cudaMemcpyAsync(imgCUDA, imgData, size, cudaMemcpyHostToDevice);

    void* buffers[] = {imgCUDA, output};

    timer.tic();
    // network forward
    tensorNet.imageInference(buffers, output_vector.size() + 1, BATCH_SIZE);
    timer.toc();
    double msTime = timer.t;
    
    std::cout << "output pointer : " <<output << std::endl;

    /*float* a = new float[100];
    cudaMemcpy(a, output, 100 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "test output 0 : " << std::endl;
    for (int i = 0; i < 100; i++)
        std::cout <<  a[i] << " ";
    std::cout << std::endl;*/

    std::cout << "label " << "    confidence " << "        xmin " << "         ymin " << "          xmax " << "        ymax" << std::endl;
    
    for (int k = 0; k < 500; k++)
    {
        if (output[7 * k + 1] == -1)
            break;
        //format: image_id, label, confidence, xmin, ymin, xmax, ymax
        float classIndex = output[7 * k + 1];
        float confidence = output[7 * k + 2];
        float xmin = output[7 * k + 3];
        float ymin = output[7 * k + 4];
        float xmax = output[7 * k + 5];
        float ymax = output[7 * k + 6];
        
        
        if (confidence > 0.7) {
            int x1 = static_cast<int>(xmin * srcImg.cols);
            int y1 = static_cast<int>(ymin * srcImg.rows);
            int x2 = static_cast<int>(xmax * srcImg.cols);
            int y2 = static_cast<int>(ymax * srcImg.rows);
            std::cout << classIndex << "         " << confidence << "        "  << x1 << "      " << y1 << "        " << x2 << "        " << y2 << std::endl;
            cv::rectangle(srcImg, cv::Rect2f(cv::Point(x1, y1), cv::Point(x2, y2)), cv::Scalar(255, 0, 255), 2);
        }
    }
    cv::imshow("output", srcImg);
    cv::waitKey(0);

    free(imgData);

    cudaFree(imgCUDA);
    cudaFree(output);

    tensorNet.destroy();

    return 0;
}
