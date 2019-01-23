#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <fstream>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"
#include "common.h"
using namespace nvuffparser;
using namespace nvinfer1;
using namespace std;

static Logger gLogger;

#define MAX_WORKSPACE (1 << 30)

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "sample_uff_lanenet: " + std::string(message);          \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)

inline int64_t volume(const Dims& d)
{
	int64_t v = 1;
	for (int64_t i = 0; i < d.nbDims; i++)
		v *= d.d[i];
	return v;
}

inline unsigned int elementSize(DataType t)
{
	switch (t)
	{
	case DataType::kINT32:
		// Fallthrough, same as kFLOAT
	case DataType::kFLOAT: return 4;
	case DataType::kHALF: return 2;
	case DataType::kINT8: return 1;
	}
	assert(0);
	return 0;
}

static const int INPUT_H = 256;
static const int INPUT_W = 512;
static const int INPUT_C = 3;

// Helper function to print blob dimensions in layers
void printTensorDims(INetworkDefinition* network,int nblayer)
{
	ILayer *layer = network -> getLayer(nblayer);
	std::cout << "Total layers: " << network->getNbLayers() << " ;#layer: " << nblayer<<" : "<< layer->getName() << " ;layer type: " << int(layer->getType()) << std::endl;
	if(int(layer->getType()) == 8){
		// 8: concatenationLayer
	}
	std::cout << "Show layer detail: " << layer->getName() << std::endl;
	std::cout << "#input tensors: " << layer->getNbInputs() <<"\t#output tensors: " << layer->getNbOutputs() <<std::endl; 

	for(int ii=0; ii < layer->getNbInputs(); ii++){
	   ITensor *tensor_in = layer-> getInput(ii);

	   std::cout << ii <<"th input tensor, Input name of tensor: " << tensor_in -> getName() << " ;dimension of the tensor: " << int(tensor_in -> getDimensions().nbDims) << std::endl;
	   for(int dd=0; dd < int(tensor_in-> getDimensions().nbDims); dd++){
		  std::cout << "\t";
		  std::cout << dd << "th dim: " << tensor_in -> getDimensions().d[dd] << std::endl;
	   }
    }
	std::cout << std::endl;

	for(int oi=0; oi < layer -> getNbOutputs(); oi++){
	   ITensor *tensor_out = layer -> getOutput(oi);

	   std::cout << oi << "th output tensor, Output name of tensor: " << tensor_out -> getName() << " ;dimension of the tensor: " << int(tensor_out -> getDimensions().nbDims) <<std::endl;
	   for(int di=0; di < int(tensor_out -> getDimensions().nbDims); di++){
          std::cout << "\t";
          std::cout << di << "th dim: " << tensor_out -> getDimensions().d[di] << std::endl;
       }
	}
	std::cout << std::endl;
}

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/lanenet/", "data/samples/lanenet/"};
    return locateFile(input,dirs);
}

void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}


std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

struct PPM
{
	std::string magic, fileName;
	int h, w, max;
	uint8_t* buffer;
//	uint8_t buffer[INPUT_C*INPUT_H*INPUT_W];
};

void* createLaneNetCudaBuffer(int64_t eltCount, DataType dtype, int run)
{
    /* in that specific case, eltCount == INPUT_H * INPUT_W * INPUT_C */
    assert(eltCount == INPUT_H * INPUT_W * INPUT_C);
    assert(elementSize(dtype) == sizeof(float));

    size_t memSize = eltCount * elementSize(dtype);
    float* inputs = new float[eltCount];

    cv::Mat rgbImage = cv::imread("./data/image.png");
    cv::Mat image;
    cv::cvtColor(rgbImage, image, CV_RGB2BGR);
    cv::resize(image, image, cv::Size(INPUT_W, INPUT_H));
    //convert to ppm just for the file format
    std::vector<PPM> ppms(1); // only one image to test

    ppms[0].buffer = image.data;
    ppms[0].h = INPUT_H;
    ppms[0].w = INPUT_W;
    std::cout<< "image read.\n";

	float* data = new float[ INPUT_C * INPUT_H * INPUT_W ];
    // watch here to restore image 
	// pixel mean 
	float pixelMean[3]{ 103.939f, 116.779f, 123.68f }; // also in BGR order
	for (int i = 0, volImg = INPUT_C*INPUT_H*INPUT_W; i < 1; ++i)
	{
		for (int c = 0; c < INPUT_C; ++c)
		{
		// the color image to input should be in BGR order
		for (unsigned j = 0, volChl = INPUT_H*INPUT_W; j < volChl; ++j)
			data[i*volImg + c*volChl + j] = float(ppms[i].buffer[j*INPUT_C + 2 - c]) - pixelMean[c];
        }
	}

    /* initialize the inputs buffer */    
    for (int i = 0; i < eltCount; i++)
        inputs[i] = float(data[i]);
    
    void* deviceMem = safeCudaMalloc(memSize);
    CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));

    delete[] inputs;
    return deviceMem;
}

void getOutput_instance_seg(int64_t eltCount, DataType dtype, void* buffer)
{
    std::cout << eltCount << " eltCount" << std::endl;
    assert(elementSize(dtype) == sizeof(float));
    std::cout << "--- OUTPUT ---" << std::endl;

    size_t memSize = eltCount * elementSize(dtype);
    float* outputs = new float[eltCount];
    CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

    cv::Mat back_image(INPUT_H, INPUT_W, CV_32FC3, cv::Scalar(255,0,0));
    int kkk = 0;
    fstream fp;
    fp.open("server_outputs.txt", ios::out);
    for(int i=0;i<INPUT_H;i++)
    {
        for(int j=0;j<INPUT_W;j++)
        {
            for(int k=0;k<(INPUT_C+1);k++)
            {
                fp << outputs[kkk] << " ";
                kkk++;
            }
        }
        fp << endl;
    }
    
    cv::imshow("test_image", back_image);
    cv::waitKey(0);
    cv::imwrite ("test.jpg ", back_image);

    float min_val_1=255, min_val_2=255, min_val_3=255;
    float max_val_1=0, max_val_2=0, max_val_3=0;
    // put output into mat and find max and min in different channel
    int iii = 0;
    for(int i=0; i < back_image.rows; i++)
    {
        for(int j=0; j < back_image.cols; j++)
        {
            back_image.at<cv::Vec3f>(i, j)[0] = outputs[iii+2];
            back_image.at<cv::Vec3f>(i, j)[1] = outputs[iii+1];
            back_image.at<cv::Vec3f>(i, j)[2] = outputs[iii];

            if(outputs[iii] < min_val_1)
                min_val_1 = outputs[iii];
            
            if(outputs[iii+1] < min_val_2)
                min_val_2 = outputs[iii+1];

            if(outputs[iii+2] < min_val_3)
                min_val_3 = outputs[iii+2];

            if(outputs[iii] > max_val_1)
                max_val_1 = outputs[iii];

            if(outputs[iii+1] > max_val_2)
                max_val_2 = outputs[iii+1];

            if(outputs[iii+2] > max_val_3)
                max_val_3 = outputs[iii+2];
            
            iii = iii + 4;
        }
    }
    // normalize into 0 - 255
    for(int i=0; i < back_image.rows; i++)
    {
        for(int j=0; j < back_image.cols; j++)
        {
            back_image.at<cv::Vec3f>(i, j)[0] = (back_image.at<cv::Vec3f>(i, j)[0] - min_val_3)*255.0/(max_val_3 - min_val_3);
            //back_image.at<cv::Vec3f>(i, j)[1] = (back_image.at<cv::Vec3f>(i, j)[1] - min_val_2)*255.0/(max_val_2-min_val_2);
            back_image.at<cv::Vec3f>(i, j)[2] = (back_image.at<cv::Vec3f>(i, j)[2] - min_val_1)*255.0/(max_val_1 - min_val_1);
        }
    }

    cv::imshow("back_image", back_image);
    cv::waitKey(0);
    cv::imwrite ("output.jpg ", back_image);
    delete[] outputs;
}

void getOutput_binary_seg(int64_t eltCount, DataType dtype, void* buffer)
{
    std::cout << eltCount << " eltCount" << std::endl;
    assert(elementSize(dtype) == sizeof(float));
    std::cout << "--- OUTPUT ---" << std::endl;

    size_t memSize = eltCount * elementSize(dtype);
    float* outputs = new float[eltCount];
    CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

    cv::Mat back_image(INPUT_H, INPUT_W, CV_32FC1, cv::Scalar(50));
    cv::imshow("gray_image", back_image);
    cv::waitKey(0);
    cv::imwrite ("gray.jpg ", back_image);

    // do argmax
    int iii = 0;
    for(int i=0; i < back_image.rows; i++)
    {
        for(int j=0; j < back_image.cols; j++)
        {
            if(outputs[iii] > outputs[iii+1])
            {
                back_image.at<float>(i, j) = 0;
            }
            else
            {
                back_image.at<float>(i, j) = 255;
            }
            iii = iii + 2;
        }
    }

    cv::imshow("final_image", back_image);
    cv::waitKey(0);
    cv::imwrite ("final.jpg ", back_image);    
    delete[] outputs;
}

ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();
    
//    int num_of_layer = 1;
//    printTensorDims(network, num_of_layer);

#if 1
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
#else
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kHALF))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
    builder->setFp16Mode(true);
#endif

    //std::cout << network->getNbLayers() << std::endl;

    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");

    /* we can clean the network and the parser */
    network->destroy();
    builder->destroy();

    return engine;
}


void execute(ICudaEngine& engine)
{
    IExecutionContext* context = engine.createExecutionContext();
    int batchSize = 1;

    int nbBindings = engine.getNbBindings();
    assert(nbBindings == 3); // attention here!

    std::vector<void*> buffers(nbBindings);
    auto buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    int bindingIdxInput = 0;
    for (int i = 0; i < nbBindings; ++i)
    {
        if (engine.bindingIsInput(i))
            bindingIdxInput = i;
        else
        {
            auto bufferSizesOutput = buffersSizes[i];
            buffers[i] = safeCudaMalloc(bufferSizesOutput.first *
                                        elementSize(bufferSizesOutput.second));
        }
    }

    auto bufferSizesInput = buffersSizes[bindingIdxInput];

    int iterations = 1;
    int numberRun = 1;
    for (int i = 0; i < iterations; i++)
    {
        float total = 0, ms;
        for (int run = 0; run < numberRun; run++)
        {
            buffers[bindingIdxInput] = createLaneNetCudaBuffer(bufferSizesInput.first,
                                                             bufferSizesInput.second, run);

            auto t_start = std::chrono::high_resolution_clock::now();
            context->execute(batchSize, &buffers[0]);
            auto t_end = std::chrono::high_resolution_clock::now();
            ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            total += ms;

            for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
            {
                if (engine.bindingIsInput(bindingIdx))
                    continue;
                
                if (bindingIdx == 1){
                    auto bufferSizesOutput = buffersSizes[bindingIdx];
                    cout << bindingIdx << endl;
                    getOutput_instance_seg(bufferSizesOutput.first, bufferSizesOutput.second, buffers[bindingIdx]);
                }

                if (bindingIdx == 2){
                    auto bufferSizesOutput = buffersSizes[bindingIdx];
                    cout << bindingIdx << endl;
                    getOutput_binary_seg(bufferSizesOutput.first, bufferSizesOutput.second, buffers[bindingIdx]);
                }
            }
            CHECK(cudaFree(buffers[bindingIdxInput]));
        }

        total /= numberRun;
        std::cout << "Average over " << numberRun << " runs is " << total << " ms." << std::endl;
    }

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        if (!engine.bindingIsInput(bindingIdx))
            CHECK(cudaFree(buffers[bindingIdx]));
    context->destroy();
}


int main(int argc, char** argv)
{
    auto fileName = locateFile("trt5_softmax.uff");
    std::cout << fileName << std::endl;

    int maxBatchSize = 1;
    auto parser = createUffParser();

    /* Register tensorflow input */
    parser->registerInput("input_tensor", DimsCHW(3, 256, 512), UffInputOrder::kNCHW);
    parser->registerOutput("MarkOutput_0");
    parser->registerOutput("MarkOutput_1");

    ICudaEngine* engine = loadModelAndCreateEngine(fileName.c_str(), maxBatchSize, parser);

    if (!engine)
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");

    /* we need to keep the memory created by the parser */
    parser->destroy();
    execute(*engine);
    engine->destroy();
    shutdownProtobufLibrary();
    
    return EXIT_SUCCESS;
}
