#include "pluginImplement.h"

#include <vector>
#include <algorithm>

/******************************/
// PluginFactory
/******************************/
bool PluginFactory::isPlugin(const char* name)
{
    return  (!strcmp(name, "conv4_3_norm")
          || !strcmp(name, "conv5_3_norm")
            
          || !strcmp(name, "conv4_3_norm_mbox_priorbox")
          || !strcmp(name, "conv5_3_norm_mbox_priorbox")
          || !strcmp(name, "fc7_mbox_priorbox")
          || !strcmp(name, "conv6_2_mbox_priorbox")

          || !strcmp(name, "detection_out")

          || !strcmp(name, "conv4_3_norm_mbox_loc_perm")
          || !strcmp(name, "conv4_3_norm_mbox_conf_perm")
          || !strcmp(name, "conv5_3_norm_mbox_loc_perm")
          || !strcmp(name, "conv5_3_norm_mbox_conf_perm")
          || !strcmp(name, "fc7_mbox_loc_perm")
          || !strcmp(name, "fc7_mbox_conf_perm")
          || !strcmp(name, "conv6_2_mbox_loc_perm")
          || !strcmp(name, "conv6_2_mbox_conf_perm")
          || !strcmp(name, "P3_mbox_loc_perm")
          || !strcmp(name, "P3_mbox_conf_perm")
          || !strcmp(name, "P4_mbox_loc_perm")
          || !strcmp(name, "P4_mbox_conf_perm")
          || !strcmp(name, "P5_mbox_loc_perm")
          || !strcmp(name, "P5_mbox_conf_perm")
          || !strcmp(name, "P6_mbox_loc_perm")
          || !strcmp(name, "P6_mbox_conf_perm")

          || !strcmp(name, "arm_loc")
          || !strcmp(name, "arm_conf")
          || !strcmp(name, "arm_priorbox")
          || !strcmp(name, "odm_loc")
          || !strcmp(name, "odm_conf")

          || !strcmp(name, "detection_out"));
}

nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights)
{
    //std::cout << layerName << std::endl;
    assert(isPlugin(layerName));
    // normalize layer
    if (!strcmp(layerName, "conv4_3_norm"))
    {
        assert(conv4_3_norm.get() == nullptr);
        conv4_3_norm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDNormalizePlugin(weights, false, false, 0.01), nvPluginDeleter);
        return conv4_3_norm.get();
    }
    if (!strcmp(layerName, "conv5_3_norm"))
    {
        assert(conv5_3_norm.get() == nullptr);
        conv5_3_norm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDNormalizePlugin(weights, false, false, 0.01), nvPluginDeleter);
        return conv5_3_norm.get();
    }

    // priorbox layer
    else if (!strcmp(layerName, "conv4_3_norm_mbox_priorbox"))
    {
        assert(conv4_3_norm_mbox_priorbox.get() == nullptr);
        PriorBoxParameters params;
        float minsize[1] = {32.0}, aspect_ratio[2] = {1.0, 2.0};
        params.minSize = minsize;
        params.aspectRatios = aspect_ratio;
        params.numMinSize = 1;
        params.numAspectRatios = 2;
        params.maxSize = NULL;
        params.numMaxSize = 0;
        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 8;
        params.stepW = 8;
        params.offset = 0.5;
        conv4_3_norm_mbox_priorbox = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return conv4_3_norm_mbox_priorbox.get();
    }
    else if (!strcmp(layerName, "conv5_3_norm_mbox_priorbox"))
    {
        assert(conv5_3_norm_mbox_priorbox.get() == nullptr);
        PriorBoxParameters params;
        float minsize[1] = {64}, aspect_ratio[2] = {1.0, 2.0};
        params.minSize = minsize;
        params.numMinSize = 1;
        params.maxSize = NULL;
        params.numMaxSize = 0;
        params.aspectRatios = aspect_ratio;
        params.numAspectRatios = 2;
        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 16;
        params.stepW = 16;
        params.offset = 0.5;
        conv5_3_norm_mbox_priorbox = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return conv5_3_norm_mbox_priorbox.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_priorbox"))
    {
        assert(fc7_mbox_priorbox.get() == nullptr);
        PriorBoxParameters params;
        float minsize[1] = {128}, aspect_ratio[2] = {1.0, 2.0};
        params.minSize = minsize;
        params.numMinSize = 1;
        params.maxSize = NULL;
        params.numMaxSize = 0;
        params.aspectRatios = aspect_ratio;
        params.numAspectRatios = 2;
        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 32;
        params.stepW = 32;
        params.offset = 0.5;
        fc7_mbox_priorbox = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return fc7_mbox_priorbox.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_priorbox"))
    {
        assert(conv6_2_mbox_priorbox.get() == nullptr);
        PriorBoxParameters params;
        float minsize[1] = {256}, aspect_ratio[2] = {1.0, 2.0};
        params.minSize = minsize;
        params.numMinSize = 1;
        params.maxSize = NULL;
        params.numMaxSize = 0;
        params.aspectRatios = aspect_ratio;
        params.numAspectRatios = 2;
        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 64;
        params.stepW = 64;
        params.offset = 0.5;
        conv6_2_mbox_priorbox = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return conv6_2_mbox_priorbox.get();
    }
    
    // detection output layer
    else if (!strcmp(layerName, "detection_out"))
    {
        assert(detection_out.get() == nullptr);
        DetectionOutputParam params;
        params.shareLocation = true;
        params.varianceEncodedInTarget = false;
        params.backgroundLabelId = 0;
        params.numClasses = 21;
        params.topK = 1000;
        params.keepTopK = 500;
        params.confidenceThreshold = 0.01;
        params.nmsThreshold = 0.45;
        params.objectnessScore = 0.01;
        params.codeType = CodeType::CENTER_SIZE;
        detection_out = std::unique_ptr<DetectionOutputPlugin>(new DetectionOutputPlugin(params));
        return detection_out.get();
    }

    // permute layer
    if (!strcmp(layerName, "conv4_3_norm_mbox_loc_perm"))
    {
        assert(conv4_3_norm_mbox_loc_perm.get() == nullptr);
        conv4_3_norm_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return conv4_3_norm_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "conv4_3_norm_mbox_conf_perm"))
    {
        assert(conv4_3_norm_mbox_conf_perm.get() == nullptr);
        conv4_3_norm_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return conv4_3_norm_mbox_conf_perm.get();
    }
    else if (!strcmp(layerName, "conv5_3_norm_mbox_loc_perm"))
    {
        assert(conv5_3_norm_mbox_loc_perm.get() == nullptr);
        conv5_3_norm_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return conv5_3_norm_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "conv5_3_norm_mbox_conf_perm"))
    {
        assert(conv5_3_norm_mbox_conf_perm.get() == nullptr);
        conv5_3_norm_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return conv5_3_norm_mbox_conf_perm.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_loc_perm"))
    {
        assert(fc7_mbox_loc_perm.get() == nullptr);
        fc7_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return fc7_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_conf_perm"))
    {
        assert(fc7_mbox_conf_perm.get() == nullptr);
        fc7_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return fc7_mbox_conf_perm.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_loc_perm"))
    {
        assert(conv6_2_mbox_loc_perm.get() == nullptr);
        conv6_2_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return conv6_2_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_conf_perm"))
    {
        assert(conv6_2_mbox_conf_perm.get() == nullptr);
        conv6_2_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return conv6_2_mbox_conf_perm.get();
    }
    else if (!strcmp(layerName, "P3_mbox_loc_perm"))
    {
        assert(P3_mbox_loc_perm.get() == nullptr);
        P3_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return P3_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "P3_mbox_conf_perm"))
    {
        assert(P3_mbox_conf_perm.get() == nullptr);
        P3_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return P3_mbox_conf_perm.get();
    }
    else if (!strcmp(layerName, "P4_mbox_loc_perm"))
    {
        assert(P4_mbox_loc_perm.get() == nullptr);
        P4_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return P4_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "P4_mbox_conf_perm"))
    {
        assert(P4_mbox_conf_perm.get() == nullptr);
        P4_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return P4_mbox_conf_perm.get();
    }
    else if (!strcmp(layerName, "P5_mbox_loc_perm"))
    {
        assert(P5_mbox_loc_perm.get() == nullptr);
        P5_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return P5_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "P5_mbox_conf_perm"))
    {
        assert(P5_mbox_conf_perm.get() == nullptr);
        P5_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return P5_mbox_conf_perm.get();
    }
    else if (!strcmp(layerName, "P6_mbox_loc_perm"))
    {
        assert(P6_mbox_loc_perm.get() == nullptr);
        P6_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return P6_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "P6_mbox_conf_perm"))
    {
        assert(P6_mbox_conf_perm.get() == nullptr);
        P6_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return P6_mbox_conf_perm.get();
    }

    // concat layer
    else if (!strcmp(layerName, "arm_loc"))
    {
        assert(arm_loc.get() == nullptr);
        arm_loc = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return arm_loc.get();
    }
    else if (!strcmp(layerName, "arm_conf"))
    {
        assert(arm_conf.get() == nullptr);
        arm_conf = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return arm_conf.get();
    }
    else if (!strcmp(layerName, "arm_priorbox"))
    {
        assert(arm_priorbox.get() == nullptr);
        arm_priorbox = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(2, true), nvPluginDeleter);
        return arm_priorbox.get();
    }
    else if (!strcmp(layerName, "odm_loc"))
    {
        assert(odm_loc.get() == nullptr);
        odm_loc = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return odm_loc.get();
    }
    else if (!strcmp(layerName, "odm_conf"))
    {
        assert(odm_conf.get() == nullptr);
        odm_conf = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        return odm_conf.get();
    }
    
    else
    {
        std::cout << "not found  " << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength)
{
    //std::cout << layerName << std::endl;
    // normalize layer
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "conv4_3_norm"))
    {
        assert(conv4_3_norm.get() == nullptr);
        conv4_3_norm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDNormalizePlugin(serialData, serialLength), nvPluginDeleter);
        return conv4_3_norm.get();
    }
    if (!strcmp(layerName, "conv5_3_norm"))
    {
        assert(conv5_3_norm.get() == nullptr);
        conv5_3_norm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDNormalizePlugin(serialData, serialLength), nvPluginDeleter);
        return conv5_3_norm.get();
    }

    // priorbox layer
    else if (!strcmp(layerName, "conv4_3_norm_mbox_priorbox"))
    {
        assert(conv4_3_norm_mbox_priorbox.get() == nullptr);
        conv4_3_norm_mbox_priorbox = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData,serialLength), nvPluginDeleter);
        return conv4_3_norm_mbox_priorbox.get();
    }
    else if (!strcmp(layerName, "conv5_3_norm_mbox_priorbox"))
    {
        assert(conv5_3_norm_mbox_priorbox.get() == nullptr);
        conv5_3_norm_mbox_priorbox = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData,serialLength), nvPluginDeleter);
        return conv5_3_norm_mbox_priorbox.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_priorbox"))
    {
        assert(fc7_mbox_priorbox.get() == nullptr);
        fc7_mbox_priorbox = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData,serialLength), nvPluginDeleter);
        return fc7_mbox_priorbox.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_priorbox"))
    {
        assert(conv6_2_mbox_priorbox.get() == nullptr);
        conv6_2_mbox_priorbox = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData,serialLength), nvPluginDeleter);
        return conv6_2_mbox_priorbox.get();
    }

    // detection out layer
    else if (!strcmp(layerName, "detection_out"))
    {
        assert(detection_out.get() == nullptr);
        detection_out = std::unique_ptr<DetectionOutputPlugin>(new DetectionOutputPlugin(serialData, serialLength));
        return detection_out.get();
    }

    // permute layer
    if (!strcmp(layerName, "conv4_3_norm_mbox_loc_perm"))
    {
        assert(conv4_3_norm_mbox_loc_perm.get() == nullptr);
        conv4_3_norm_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return conv4_3_norm_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "conv4_3_norm_mbox_conf_perm"))
    {
        assert(conv4_3_norm_mbox_conf_perm.get() == nullptr);
        conv4_3_norm_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return conv4_3_norm_mbox_conf_perm.get();
    }
    else if (!strcmp(layerName, "conv5_3_norm_mbox_loc_perm"))
    {
        assert(conv5_3_norm_mbox_loc_perm.get() == nullptr);
        conv5_3_norm_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return conv5_3_norm_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "conv5_3_norm_mbox_conf_perm"))
    {
        assert(conv5_3_norm_mbox_conf_perm.get() == nullptr);
        conv5_3_norm_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return conv5_3_norm_mbox_conf_perm.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_loc_perm"))
    {
        assert(fc7_mbox_loc_perm.get() == nullptr);
        fc7_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return fc7_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "fc7_mbox_conf_perm"))
    {
        assert(fc7_mbox_conf_perm.get() == nullptr);
        fc7_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return fc7_mbox_conf_perm.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_loc_perm"))
    {
        assert(conv6_2_mbox_loc_perm.get() == nullptr);
        conv6_2_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return conv6_2_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "conv6_2_mbox_conf_perm"))
    {
        assert(conv6_2_mbox_conf_perm.get() == nullptr);
        conv6_2_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return conv6_2_mbox_conf_perm.get();
    }
    else if (!strcmp(layerName, "P3_mbox_loc_perm"))
    {
        assert(P3_mbox_loc_perm.get() == nullptr);
        P3_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return P3_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "P3_mbox_conf_perm"))
    {
        assert(P3_mbox_conf_perm.get() == nullptr);
        P3_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return P3_mbox_conf_perm.get();
    }
    else if (!strcmp(layerName, "P4_mbox_loc_perm"))
    {
        assert(P4_mbox_loc_perm.get() == nullptr);
        P4_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return P4_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "P4_mbox_conf_perm"))
    {
        assert(P4_mbox_conf_perm.get() == nullptr);
        P4_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return P4_mbox_conf_perm.get();
    }
    else if (!strcmp(layerName, "P5_mbox_loc_perm"))
    {
        assert(P5_mbox_loc_perm.get() == nullptr);
        P5_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return P5_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "P5_mbox_conf_perm"))
    {
        assert(P5_mbox_conf_perm.get() == nullptr);
        P5_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return P5_mbox_conf_perm.get();
    }
    else if (!strcmp(layerName, "P6_mbox_loc_perm"))
    {
        assert(P6_mbox_loc_perm.get() == nullptr);
        P6_mbox_loc_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return P6_mbox_loc_perm.get();
    }
    else if (!strcmp(layerName, "P6_mbox_conf_perm"))
    {
        assert(P6_mbox_conf_perm.get() == nullptr);
        P6_mbox_conf_perm = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return P6_mbox_conf_perm.get();
    }
    
    // concat layer
    else if (!strcmp(layerName, "arm_loc"))
    {
        assert(arm_loc.get() == nullptr);
        arm_loc = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData,serialLength), nvPluginDeleter);
        return arm_loc.get();
    }
    else if (!strcmp(layerName, "arm_conf"))
    {
        assert(arm_conf.get() == nullptr);
        arm_conf = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData,serialLength), nvPluginDeleter);
        return arm_conf.get();
    }
    else if (!strcmp(layerName, "arm_priorbox"))
    {
        assert(arm_priorbox.get() == nullptr);
        arm_priorbox = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData,serialLength), nvPluginDeleter);
        return arm_priorbox.get();
    }
    else if (!strcmp(layerName, "odm_loc"))
    {
        assert(odm_loc.get() == nullptr);
        odm_loc = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData,serialLength), nvPluginDeleter);
        return odm_loc.get();
    }
    else if (!strcmp(layerName, "odm_conf"))
    {
        assert(odm_conf.get() == nullptr);
        odm_conf = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData,serialLength), nvPluginDeleter);
        return odm_conf.get();
    }

    else
    {
        std::cout << "not found" << std::endl;
        assert(0);
        return nullptr;
    }
}

void PluginFactory::destroyPlugin()
{   
    //normalize layer
    conv4_3_norm.release();
    conv4_3_norm = nullptr;
    conv5_3_norm.release();
    conv5_3_norm = nullptr;

    //priorbox layer
    conv4_3_norm_mbox_priorbox.release();
    conv4_3_norm_mbox_priorbox = nullptr;
    conv5_3_norm_mbox_priorbox.release();
    conv5_3_norm_mbox_priorbox = nullptr;
    fc7_mbox_priorbox.release();
    fc7_mbox_priorbox = nullptr;
    conv6_2_mbox_priorbox.release();
    conv6_2_mbox_priorbox = nullptr;

    //permute layer
    conv4_3_norm_mbox_loc_perm.release();
    conv4_3_norm_mbox_loc_perm = nullptr;
    conv4_3_norm_mbox_conf_perm.release();
    conv4_3_norm_mbox_conf_perm = nullptr;
    conv5_3_norm_mbox_loc_perm.release();
    conv5_3_norm_mbox_loc_perm = nullptr;
    conv5_3_norm_mbox_conf_perm.release();
    conv5_3_norm_mbox_conf_perm = nullptr;
    fc7_mbox_loc_perm.release();
    fc7_mbox_loc_perm = nullptr;
    fc7_mbox_conf_perm.release();
    fc7_mbox_conf_perm = nullptr;
    conv6_2_mbox_loc_perm.release();
    conv6_2_mbox_loc_perm = nullptr;
    conv6_2_mbox_conf_perm.release();
    conv6_2_mbox_conf_perm = nullptr;
    P3_mbox_loc_perm.release();
    P3_mbox_loc_perm = nullptr;
    P3_mbox_conf_perm.release();
    P3_mbox_conf_perm = nullptr;
    P4_mbox_loc_perm.release();
    P4_mbox_loc_perm = nullptr;
    P4_mbox_conf_perm.release();
    P4_mbox_conf_perm = nullptr;
    P5_mbox_loc_perm.release();
    P5_mbox_loc_perm = nullptr;
    P5_mbox_conf_perm.release();
    P5_mbox_conf_perm = nullptr;
    P6_mbox_loc_perm.release();
    P6_mbox_loc_perm = nullptr;
    P6_mbox_conf_perm.release();
    P6_mbox_conf_perm = nullptr;

    // concat layer
    arm_loc.release();
    arm_loc = nullptr;
    arm_conf.release();
    arm_conf = nullptr;
    arm_priorbox.release();
    arm_priorbox = nullptr;
    odm_loc.release();
    odm_loc = nullptr;
    odm_conf.release();
    odm_conf = nullptr;

    // detection out layer
    detection_out.release();
    detection_out = nullptr;
}
