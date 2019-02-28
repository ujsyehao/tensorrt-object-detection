#ifndef __PLUGIN_LAYER_H__
#define __PLUGIN_LAYER_H__

#include <iostream>
#include <cassert>
#include <cstring>
#include <memory>

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

#include "detection_output.h"

#define CHECK(status)                                                                                           \
    {                                                                                                                           \
        if (status != 0)                                                                                                \
        {                                                                                                                               \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) \
                      << " at line " << __LINE__                                                        \
                      << std::endl;                                                                     \
            abort();                                                                                                    \
        }                                                                                                                               \
    }

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override;
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;

    void(*nvPluginDeleter)(INvPlugin*) { [](INvPlugin* ptr) {ptr->destroy(); } };

    bool isPlugin(const char* name) override;
    void destroyPlugin();

    //normalize layer
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv5_3_norm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv4_3_norm{ nullptr, nvPluginDeleter };

    //priorbox layers
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv4_3_norm_mbox_priorbox{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv5_3_norm_mbox_priorbox{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> fc7_mbox_priorbox{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv6_2_mbox_priorbox{ nullptr, nvPluginDeleter };

    //detection output layer
    std::unique_ptr<DetectionOutputPlugin> detection_out{ nullptr };

    //permute layers
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv4_3_norm_mbox_loc_perm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv4_3_norm_mbox_conf_perm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv5_3_norm_mbox_loc_perm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv5_3_norm_mbox_conf_perm{ nullptr, nvPluginDeleter };
	std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> fc7_mbox_loc_perm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> fc7_mbox_conf_perm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv6_2_mbox_loc_perm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> conv6_2_mbox_conf_perm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> P3_mbox_loc_perm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> P3_mbox_conf_perm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> P4_mbox_loc_perm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> P4_mbox_conf_perm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> P5_mbox_loc_perm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> P5_mbox_conf_perm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> P6_mbox_loc_perm{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> P6_mbox_conf_perm{ nullptr, nvPluginDeleter };

    //concat layers
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> arm_loc{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> arm_conf{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> arm_priorbox{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> odm_loc{ nullptr, nvPluginDeleter };
    std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> odm_conf{ nullptr, nvPluginDeleter };
    
};

#endif
