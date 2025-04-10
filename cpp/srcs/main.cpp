#include <fstream>
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

    std::ifstream file(model_path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open engine file");
    std::cout << "Engine file loaded." << std::endl;

    // seek the end of the file to determine engine size
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    // create a file to hold engine data
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    // Create TensorRT runtime - to deserealize model and process inference
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime)
    {
        std::cerr << "Error: Failed to create TensorRT runtime" << std::endl;
        return -1;
    }

    // Deserialize engine
    nvinfer1::ICudaEngine* engine =
        runtime->deserializeCudaEngine(engineData.data(), size);
    if (!engine)
    {
        std::cerr << "Error: Failed to deserialize engine" << std::endl;
        return -1;
    }
    std::cout << "Engine loaded successfully." << std::endl;

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context)
        throw std::runtime_error("Failed to create execution context.");
    std::cout << "Execution context created." << std::endl;

    checkEngineSpecs(engine);

    // WARN: these should be size in byte!! Yes we get 256 x 256 element in the
    // output, but storing floats, so the size in byte should be 256 * 256 *
    // sizeof(float)
    size_t inputSize = 196608;
    size_t outputSize = 65536 * sizeof(float);

    // Allocate memory for input tensor (binding index 0)
    void* inputDevice;
    cudaError_t status = cudaMalloc(&inputDevice, inputSize);
    if (status != cudaSuccess)
        throw std::runtime_error("Failed to allocate device memory for input.");

    // Allocate memory for output tensor (binding index 1)
    void* outputDevice;
    status = cudaMalloc(&outputDevice, outputSize);
    if (status != cudaSuccess)
        throw std::runtime_error(
            "Failed to allocate device memory for output.");

    std::vector<float> og_image;
    og_image = loadImage(img_path);

    // Copy data from host to device for the input
    status = cudaMemcpy(inputDevice, og_image.data(), inputSize,
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
        throw std::runtime_error(
            "Failed to copy data from host to device for input.");
    std::cout << "Data copied from host to device." << std::endl;

    // Set up the bindings
    void* bindings[2]; // assuming 2 bindings: input and output

    // Bind input
    bindings[0] = inputDevice; // inputDevice holds the input data (image)

    // Bind output
    bindings[1] = outputDevice; // outputDevice will hold the result

    // Run the inference
    bool context_status = context->executeV2(bindings);
    if (!context_status)
        throw std::runtime_error("Failed to execute inference.");
    std::cout << "Inference executed successfully." << std::endl;

    // Copy data from device to host for the output
    std::vector<float> outputHostData(outputSize / sizeof(float));
    status = cudaMemcpy(outputHostData.data(), outputDevice,
                        outputSize / sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
        throw std::runtime_error(
            "Failed to copy data from device to host for output.");

    // Reshape the output into a 2D matrix (same size as input image)
    int outputHeight = 256; // Height of the output mask (same as input image)
    int outputWidth = 256;  // Width of the output mask (same as input image)
    std::vector<float> outputMask(outputHeight * outputWidth);

    // Copy the output data into the mask
    for (int i = 0; i < outputHeight * outputWidth; ++i)
    {
        outputMask[i] =
            outputHostData[i]; // Assuming outputHostData is flattened
    }

    std::cout << "Output buffer size: " << outputHostData.size() << std::endl;

    // Apply threshold to the mask (qualquer coisa acima do threshold e' mask,
    // abaixo e' background)
    // float threshold = 0.5f; // You can experiment with this value ->
    //                         // experimentei, resultados muito semelhantes
    // for (int i = 0; i < outputHeight * outputWidth; ++i)
    // {
    //     if (outputMask[i] >= threshold)
    //     {
    //         outputMask[i] = 1.0f; // Lane
    //     }
    //     else
    //     {
    //         outputMask[i] = 0.0f; // background
    //     }
    // }

    // Convert the output mask to an OpenCV Mat
    cv::Mat laneMask(outputHeight, outputWidth, CV_32F,
                     outputMask.data()); // CV_32F for float

    // Normalize the mask to [0, 255] (since OpenCV expects 8-bit or 32-bit)
    cv::Mat displayMask;
    laneMask.convertTo(displayMask, CV_8U,
                       255.0); // Convert to 8-bit, 255 for white

    // Save or display the mask
    cv::imwrite("results/lane_mask.jpg", displayMask);

    cudaFree(inputDevice);
    cudaFree(outputDevice);
    delete context;
    delete engine;
    delete runtime;
    std::cout << "Process completed" << std::endl;
    return 0;
}
