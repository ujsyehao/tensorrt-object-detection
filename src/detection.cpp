#include "detection.h"

//int Detect::BATCH_SIZE_ = 1;
void ImgPreprocess(cv::Mat &input, int re_width, int re_height, float *data_unifrom, const float3 mean, const float scale)
{
    int line_offset;
    int offset_g;
    int offset_r;
    
    unsigned char *line = NULL;
    float *unifrom_data = data_unifrom;

    cv::Mat dst;
    cv::resize(input, dst, cv::Size(re_width, re_height), (0.0), (0.0), cv::INTER_LINEAR );
    offset_g = re_width * re_height;
    offset_r = re_width * re_height * 2;
    for (int i = 0; i < re_height; ++i)
    {
        line = dst.ptr<uchar>(i);
        line_offset = i * re_width;
        for(int j = 0; j < re_width; ++j)
        {
            // b
            unifrom_data[line_offset + j] =  float(line[j * 3] - mean.x);
            // g
            unifrom_data[offset_g + line_offset + j] = float(line[j * 3 + 1] - mean.y);
            // r
            unifrom_data[offset_r + line_offset + j] = float(line[j * 3 + 2] - mean.z);
        }
    }  
}

Detect::Detect(const char* model, const char* weight, int model_width, int model_height, float threshold)
{
    // initial value
    model_ = model;
    weight_ = weight;
    std::string input = "data";
    std::string output = "detection_out";
    INPUT_BLOB_NAME_ = input.c_str();
    OUTPUT_BLOB_NAME_ = output.c_str();
    //BATCH_SIZE_ = 1;

    // get model input size
    width_ = model_width;
    height_ = model_height;

    // get threshold
    threshold_ = threshold;

    // create network
    std::vector<std::string> output_vector = {OUTPUT_BLOB_NAME_};
	tensorNet_ = new TensorNet();
    tensorNet_->LoadNetwork(model_, weight_, INPUT_BLOB_NAME_, output_vector, BATCH_SIZE_);

    // allocate input image CPU memory -> Initialize to 0
    size_ = width_ * height_ * 3 * sizeof(float);
    imgData_ = malloc(size_);
    memset(imgData_, 0, size_);       

    // allocate input image GPU memory -> Initialize to 0
    cudaMalloc(&imgCUDA_, size_);
    cudaMemset(imgCUDA_, 0, size_);

    //  allocate output memory -> Initialize to 0   
    cudaMallocManaged(&output_, 1 * 110 * 7 * sizeof(float));
    memset(output_, 0, 1 * 110 * 7 * sizeof(float));
}

void Detect::ImgInference(cv::Mat &img)
{
    // note: avoid last frame detection result cover current frame detection result
	memset(output_, 0, 1 * 110 * 7 * sizeof(float));

    cv::Mat srcImg = img.clone();
    cv::Mat frame = img;

    // pre-process
    ImgPreprocess(frame, height_, width_, (float*)imgData_, make_float3(104, 117, 123), 1);

    // copy img from cpu to gpu
    cudaMemcpy(imgCUDA_, imgData_, size_, cudaMemcpyHostToDevice);

    // model forward 
    void* buffers[] = {imgCUDA_, output_};
    Timer timer;
    timer.tic();
    tensorNet_->imageInference(buffers, 2, BATCH_SIZE_);

    // parse outputs
    for (int k = 0; k < 110; k++)
    {
        //format: image_id, label, confidence, xmin, ymin, xmax, ymax
        if (output_[7 * k] == 0) {
            //std::cout << "the " << k << "th bbox break " << std::endl;                
            break;
        }
        
        float classIndex = output_[7 * k + 1];
        float confidence = output_[7 * k + 2];
        float xmin = output_[7 * k + 3];
        float ymin = output_[7 * k + 4];
        float xmax = output_[7 * k + 5];
        float ymax = output_[7 * k + 6];
        //std::cout << classIndex << " " << confidence << " " << xmin << " " << ymin << " " << xmax << " " << ymax << std::endl;
                    
        if (confidence > threshold_) {
            int x1 = static_cast<int>(xmin * srcImg.cols);
            int y1 = static_cast<int>(ymin * srcImg.rows);
            int x2 = static_cast<int>(xmax * srcImg.cols);
            int y2 = static_cast<int>(ymax * srcImg.rows); 
            int width = x2 - x1;
            int height = y2 - y1;
			char txt[64];
			sprintf(txt, "%.2f", confidence);
            cv::rectangle(img, cv::Rect2f(cv::Point(x1, y1), cv::Point(x2, y2)), cv::Scalar(0, 255, 255), 2);
			//cv::putText(img, std::to_string(confidence), cv::Point(x1, y1), 2, cv::Scalar(0, 255, 255), 1, 8);
			cv::putText(img, txt, cv::Point(x1, y1), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 1, 3, 0);
            //std::cout << confidence << " " << x1 << " " << y1 << " " << width << " " << height << std::endl;                
        }
    }      
    timer.toc();       
    double msTime = timer.t;
    std::cout << "forward time + post-process time(nms + confidence): " << msTime << "ms" << std::endl;   
}

Detect::~Detect()
{
	//std::cout << "fuck" << std::endl;
    free(imgData_);
    imgData_ = NULL;

    cudaFree(imgCUDA_);
    imgCUDA_ = NULL;
    cudaFree(output_);
    output_ = NULL;
    
    tensorNet_->destroy();    
}
