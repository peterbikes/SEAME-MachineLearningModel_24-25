#include <iostream>
#include <utils.hpp>
#include <vector>

int main(int argc, char** argv)
{
    Logger logger;

    if (argc < 2)
    {
        std::cerr << "provide a path to a picture as argument\n";
        return -1;
    }
    std::string img_path(argv[1]);

    // Create TensorRT runtime - to deserealize model and process inference
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime)
    {
        std::cerr << "Error: Failed to create TensorRT runtime" << std::endl;
        return -1;
    }

    nvinfer1::ICudaEngine* engine = createEngine(runtime);
    if (!engine)
    {
        std::cerr << "Error: Failed to deserialize engine" << std::endl;
        delete runtime;
        return -1;
    }
    std::cout << "Engine loaded successfully." << std::endl;

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context)
    {
        std::cerr << "Error: Failed to create context\n";
        delete runtime;
        delete engine;
        return -1;
    }
    std::cout << "Execution context created." << std::endl;

    checkEngineSpecs(engine);

    // Load and preprocess the image
    std::vector<float> input_data = loadImage(img_path);

    // Get binding indices
    const int inputIndex = engine->getBindingIndex("input_1");
    const int outputIndex = engine->getBindingIndex("conv2d_14");

    // Get input and output dimensions
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);

    // WARN: Calculate buffer sizes : the size of the input and output buffer
    // must be dynamically calulated using binding dimension.... ex: input
    // tensor is 1 256 256 3 (NHWC). its size has to be N * H * W * C *
    // sizeof(float) : 256 * 256 element per channel , 3 channel, each element
    // beeing a float32
    size_t inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; i++)
        inputSize *= inputDims.d[i];
    inputSize *= sizeof(float);

    size_t outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; i++)
        outputSize *= outputDims.d[i];
    outputSize *= sizeof(float);

    std::cout << "Input size: " << inputSize << " bytes" << std::endl;
    std::cout << "Output size: " << outputSize << " bytes" << std::endl;

    // Allocate GPU memory for input and output
    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    // Copy input data to GPU
    cudaMemcpy(d_input, input_data.data(), inputSize, cudaMemcpyHostToDevice);

    // Create GPU buffers array
    void* bindings[2] = {d_input, d_output};

    // Execute inference
    bool status = context->executeV2(bindings);
    if (!status)
    {
        std::cerr << "Error during inference execution" << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        delete context;
        delete engine;
        delete runtime;
        return -1;
    }

    // Allocate CPU memory for output
    std::vector<float> output_data(outputSize / sizeof(float));

    // Copy output data from GPU to CPU
    cudaMemcpy(output_data.data(), d_output, outputSize,
               cudaMemcpyDeviceToHost);

    // Convert to image
    int width = 256;
    int height = 256;
    cv::Mat output_image(height, width, CV_32FC1, output_data.data());

    // Try several operations and save the result
    debugOutput(output_data, output_image);

    cudaFree(d_input);
    cudaFree(d_output);
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
