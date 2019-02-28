#ifndef DETECTION_OUTPUT_H_
#define DETECTION_OUTPUT_H_

#include <assert.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

typedef enum
{
    CORNER      = 1,
    CENTER_SIZE = 2,
    CORNER_SIZE = 3
} CodeType;

struct DetectionOutputParam
{
    bool shareLocation, varianceEncodedInTarget;
    int backgroundLabelId, numClasses, topK, keepTopK;
    float confidenceThreshold, nmsThreshold, objectnessScore;
    CodeType codeType;
};

class DetectionOutputPlugin : public IPluginExt {
public:
    DetectionOutputPlugin(const DetectionOutputParam& param);
    DetectionOutputPlugin(const void* buffer, size_t size);
    ~DetectionOutputPlugin();

    virtual int getNbOutputs() const override;
    virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    virtual bool supportsFormat(DataType type, PluginFormat format) const override;
    virtual void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims,
                                     int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;

    virtual int initialize() override;
    virtual void terminate() override;

    virtual size_t getWorkspaceSize(int maxBatchSize) const override;
    virtual size_t getSerializationSize() override;

    virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;
    virtual void serialize(void* buffer) override;

private:
    template <typename T>
    void read(const char*& buffer, T& val) 
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }

    template <typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    /*size_t type2size(DataType type)
    {
        //return type == DataType::KFLOAT ? sizeof(float) : sizeof(__half);
        return sizeof(float);
    }*/

private:
    /*int num_classes_; 
    bool share_location_; 
    int background_label_id_; 
    CodeType code_type_; 
    bool variance_encoded_in_target_; 
    int keep_top_k_; 
    float confidence_threshold_; 
    float nums_threshold_; 
    float objectness_score_;
    int top_k_; */
    DetectionOutputParam params;

    int num_loc_classes_;

    // unused
    //int num_;

    // prior boxs number
    int num_priors_;

    float eta_;
};
#endif