#include "tensorNet.h"

#include <algorithm>
#include <sstream>
#include <fstream>

using namespace nvinfer1;

bool TensorNet::LoadNetwork(const char* prototxt_path,
                            const char* model_path,
                            const char* input_blob,
                            const std::vector<std::string>& output_blobs,
                            uint32_t maxBatchSize)
{
    // attempt to load network from cache before profiling with tensorRT
    std::stringstream gieModelStdStream;
    gieModelStdStream.seekg(0, gieModelStdStream.beg);
    char cache_path[512];
    sprintf(cache_path, "%s.plan", model_path);
	//sprintf(cache_path, "%s", model_path);
    
    std::string tmp_model(model_path);
    std::string match_model("caffemodel");
    std::size_t found = tmp_model.find(match_model);
    if (found != std::string::npos) 
    {
        std::cout << "it is a caffemodel" << std::endl;
        std::string tail = tmp_model.substr(0, tmp_model.size() - 10);
        //std::cout << tmp_model << std::endl;
        //std::cout << tail << std::endl;
        sprintf(cache_path, "%splan", tail.c_str());
        //std::cout << cache_path << std::endl;
    } 
    else 
    {
        std::cout << "it is a plan" << std::endl;
        sprintf(cache_path, "%s", model_path);
    }

	printf( "attempting to open cache file %s\n", cache_path);

    std::ifstream cache(cache_path);

    if(!cache)
    {
        printf( "cache file not found, profiling network model\n");

        if( !caffeToTRTModel(prototxt_path, model_path, output_blobs, maxBatchSize, gieModelStdStream) )
        {
            printf("failed to load %s\n", model_path);
            return 0;
        }
        printf( "network profiling complete, writing cache to %s\n", cache_path);
        std::ofstream outFile;
        outFile.open(cache_path);
        outFile << gieModelStdStream.rdbuf();
        outFile.close();
        gieModelStdStream.seekg(0, gieModelStdStream.beg);
        printf( "completed writing cache to %s\n", cache_path);

        infer = createInferRuntime(gLogger);
        /**
         * deserializeCudaEngine can be used to load the serialized CuDA Engine (Plan file).
         * */
        std::cout << "size1: " << gieModelStream->size() << std::endl;
        std::cout << "createInference" << std::endl;
        engine = infer->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);
        context = engine->createExecutionContext(); // allocate device memory
        std::cout << "createInference_end" << std::endl;
        printf("Bindings after deserializing:\n");
        for (int bi = 0; bi < engine->getNbBindings(); bi++) {
            if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
            else printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
        }
        gieModelStream->destroy();
    }
    else
    {
        std::cout << "loading network profile from cache..." << std::endl;
        gieModelStdStream << cache.rdbuf();
        cache.close();
        gieModelStdStream.seekg(0, std::ios::end);
        const int modelSize = gieModelStdStream.tellg();
        //std::cout << "model size: " << modelSize << std::endl;
        gieModelStdStream.seekg(0, std::ios::beg);
        void* modelMem = malloc(modelSize);
        gieModelStdStream.read((char*)modelMem, modelSize);

        infer = createInferRuntime(gLogger);
        //std::cout << "createInference" << std::endl;
        engine = infer->deserializeCudaEngine(modelMem, modelSize, &pluginFactory);
        context = engine->createExecutionContext(); // allocate device memory
        //free(modelMem);
        std::cout << "createInference_end" << std::endl;
        //printf("Bindings after deserializing:\n");
        /*for (int bi = 0; bi < engine->getNbBindings(); bi++) {
            if (engine->bindingIsInput(bi) == true) 
                printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
            else 
                printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
        }*/
    }   
}

bool TensorNet::caffeToTRTModel(const char* deployFile,
                                const char* modelFile,
                                const std::vector<std::string>& outputs,
                                unsigned int maxBatchSize,
                                std::ostream& gieModelStdStream)
{
    // create the builder and network
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    // create the caffe parser
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactory(&pluginFactory);

    bool useFp16 = builder->platformHasFastFp16();

    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT;

    std::cout << "use FP16 " << useFp16 << std::endl;
    std::cout << deployFile <<std::endl;
    std::cout << modelFile <<std::endl;

    // parse the model
    const IBlobNameToTensor* blobNameToTensor =	parser->parse(deployFile,
                                                              modelFile,
                                                              *network,
                                                              modelDataType);

    assert(blobNameToTensor != nullptr);

    std::cout << "finish parsing model" << std::endl;


    // test layer output code
    /*const char* output = "conv4_3_norm";
    DimsCHW out = getTensorDims(output);
    std::cout << "conv1_1 shape:" << out.c() << " " << out.h() << " " << out.w() << std::endl;*/

    // specify the outputs of network
    for (auto& s : outputs) 
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // create a optimized runtime/engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 << 20);

    if (useFp16)
    {
        builder->setHalf2Mode(true);
    }

    // build the engine -> copy weights
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // destroy the network and the parser
    network->destroy();
    parser->destroy();

    // serialize the engine -> for inference
    gieModelStream = engine->serialize();
    if(!gieModelStream)
    {
        std::cout << "failed to serialize CUDA engine" << std::endl;
        return false;
    }

    //std::cout << "size0: " << gieModelStream->size() << std::endl;

    // store model
    gieModelStdStream.write((const char*)gieModelStream->data(), gieModelStream->size());

    // destroy the IHostMemory*, builder, engine,  pluginFactory
    //gieModelStream->destroy();
    builder->destroy();
    engine->destroy();
    pluginFactory.destroyPlugin();
    shutdownProtobufLibrary();

    std::cout << "caffeToTRTModel Finished" << std::endl;

    return true;
}

/**
 * This function de-serializes the cuda engine.
 * */
void TensorNet::createInference()
{
    infer = createInferRuntime(gLogger);
    /**
     * deserializeCudaEngine can be used to load the serialized CuDA Engine (Plan file).
     * */
    engine = infer->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);

    printf("Bindings after deserializing:\n");
    for (int bi = 0; bi < engine->getNbBindings(); bi++) {
        if (engine->bindingIsInput(bi) == true) 
            printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
        else 
            printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
    }
}

void TensorNet::imageInference(void** buffers, int nbBuffer, int batchSize)
{
    //std::cout << "Came into the image inference method here. " << std::endl;

    assert(engine->getNbBindings() == nbBuffer);
     
    //context = engine->createExecutionContext(); // allocate device memory
    // tensorRT asynchronous execution
    context->setProfiler(&gProfiler);

    //std::cout << "inference batchsize: " << batchSize << std::endl;

    // buffers: an array of pointers to input and output buffers for the network
    bool status = context->execute(batchSize, buffers);
    //std::cout << "status: " << status << std::endl;

    //std::cout << "####################" << std::endl;
    //context->destroy();
}

void TensorNet::timeInference(int iteration, int batchSize)
{
    int inputIdx = 0;
    size_t inputSize = 0;
    void* buffers[engine->getNbBindings()];

    for (int b = 0; b < engine->getNbBindings(); b++)
    {
        DimsCHW dims = static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
        size_t size = batchSize * dims.c() * dims.h() * dims.w() * sizeof(float);
        CHECK(cudaMalloc(&buffers[b], size));

        if(engine->bindingIsInput(b) == true)
        {
            inputIdx = b;
            inputSize = size;
        }
    }

    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);

    CHECK(cudaMemset(buffers[inputIdx], 0, inputSize));

    for (int i = 0; i < iteration;i++) context->execute(batchSize, buffers);

    context->destroy();
    for (int b = 0; b < engine->getNbBindings(); b++) CHECK(cudaFree(buffers[b]));

}

DimsCHW TensorNet::getTensorDims(const char* name)
{
    //return static_cast<DimsCHW&&>(engine->getBindingDimensions(1));
    
    //std::cout << name << std::endl;
    //std::cout << "___________________" << std::endl;
    //std::cout << engine->getNbBindings() << std::endl;
    //std::cout << "!!!!!!!!!!!!!!!!!" << std::endl;
    for (int b = 0; b < engine->getNbBindings(); b++) {
        //std::cout << "b value: " << b << std::endl;
        if( !strcmp( name, engine->getBindingName(b))) {
            //std::cout << "engine->getBindingName: " << engine->getBindingName(b) << std::endl;
            return static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
        }
    }
    return DimsCHW{0,0,0};
}

void TensorNet::printTimes(int iteration)
{
    gProfiler.printLayerTimes(iteration);
}

void TensorNet::destroy()
{
	std::cout << "destroy tensorrt" << std::endl;
    context->destroy();
    pluginFactory.destroyPlugin();
    engine->destroy();
    infer->destroy();
}
