#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

const std::string model_path = "correct.engine";

class Logger : public nvinfer1::ILogger
{
    public:
        void log(Severity severity, const char* msg) noexcept override
        {
            std::cerr << "[TensorRT] ";
            switch (severity)
            {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "[ERROR] ";
                break;
            case Severity::kERROR:
                std::cerr << "[ERROR] ";
                break;
            case Severity::kWARNING:
                std::cerr << "[WARNING] ";
                break;
            case Severity::kINFO:
                std::cerr << "[INFO] ";
                break;
            case Severity::kVERBOSE:
                std::cerr << "[VERBOSE] ";
                break;
            default:
                break;
            }
            std::cerr << msg << std::endl;
        }
} gLogger;

// move this function to utils:
void checkEngineSpecs(nvinfer1::ICudaEngine* engine)
{
    int numBindings = engine->getNbBindings();
    std::cout << "Number of bindings: " << numBindings << std::endl;

    for (int i = 0; i < numBindings; ++i)
    {
        const char* name = engine->getBindingName(i);
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        bool isInput = engine->bindingIsInput(i);

        std::cout << "Binding index " << i << ": " << name << std::endl;
        std::cout << "  Is input: " << (isInput ? "Yes" : "No") << std::endl;
        std::cout << "  Dimensions: ";
        for (int j = 0; j < dims.nbDims; ++j)
        {
            std::cout << dims.d[j] << " ";
        }
        std::cout << std::endl;
        std::cout << "  Data type: "
                  << (dtype == nvinfer1::DataType::kFLOAT ? "FLOAT" : "Other")
                  << std::endl;
    }
}

std::vector<float> loadImage()
{
    // IMAGE LOADING/RESIZING PART --> this is now a function
    fs::create_directory("results");
    std::string image_path = "images/dark_frame_0242.jpg";
    cv::imwrite("results/original_pic.jpg", cv::imread(image_path));

    // Load color image
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        throw std::runtime_error("Failed to load image");
    }
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;

    // Resize image to 256x256
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(256, 256));
    std::cout << "Resized image size: " << resized_img.size() << std::endl;
    cv::imwrite("results/resized_image.jpg", resized_img);

    // Preparing vector where image data will be stored
    std::vector<float> og_image(256 * 256 * 3);

    // Fill vector with pixel values (normalized)
    for (int y = 0; y < resized_img.rows; ++y)
    {
        for (int x = 0; x < resized_img.cols; ++x)
        {
            cv::Vec3b pixel = resized_img.at<cv::Vec3b>(y, x);
            // Store the pixel data in the og_image vector (RGB channels) -->
            // normalized by dividing by 255
            og_image[(y * resized_img.cols + x) * 3] =
                static_cast<float>(pixel[0]) / 255.0f; // Red channel
            og_image[(y * resized_img.cols + x) * 3 + 1] =
                static_cast<float>(pixel[1]) / 255.0f; // Green channel
            og_image[(y * resized_img.cols + x) * 3 + 2] =
                static_cast<float>(pixel[2]) / 255.0f; // Blue channel
        }
    }

    // Create a cv::Mat from the normalized image data --- > only to see it
    cv::Mat normalizedImg(resized_img.rows, resized_img.cols, CV_32FC3,
                          og_image.data());
    // Convert the image back to a displayable format (0-255 range)
    cv::Mat displayImg;
    normalizedImg.convertTo(displayImg, CV_8UC3, 255.0);
    // Save or show the image
    cv::imwrite("results/normalized_image.jpg", displayImg);

    return og_image;
}

int main()
{
    // Load the model (TensorRT engine) -- > turn into function ??
    std::ifstream file(model_path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open engine file");
    }
    std::cout << "Engine file loaded." << std::endl;

    // seek the end of the file to determine engine size
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    // create a file to hold engine data
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    // Create TensorRT runtime - to deserealize model and process inference
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
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

    // Create execution context
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context)
    {
        throw std::runtime_error("Failed to create execution context.");
    }
    std::cout << "Execution context created." << std::endl;

    // engine info test function below
    checkEngineSpecs(engine);

    // checkInputDimensions(engine);
    size_t inputSize =
        196608; // ->create func for both of this with the information above
    size_t outputSize = 65536;

    // Allocate memory for input tensor (binding index 0)
    void* inputDevice;
    cudaError_t status = cudaMalloc(&inputDevice, inputSize);
    if (status != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate device memory for input.");
    }

    // Allocate memory for output tensor (binding index 1)
    void* outputDevice;
    status = cudaMalloc(&outputDevice, outputSize);
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            "Failed to allocate device memory for output.");
    }

    std::vector<float> og_image;
    og_image = loadImage();

    // Copy data from host to device for the input
    status = cudaMemcpy(inputDevice, og_image.data(), inputSize,
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            "Failed to copy data from host to device for input.");
    }
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
    {
        throw std::runtime_error("Failed to execute inference.");
    }
    std::cout << "Inference executed successfully." << std::endl;

    // a partir daqui, o modelo ja correu, vamos retirar a informacao do output
    // para uma imagem

    // Copy data from device to host for the output
    std::vector<float> outputHostData(outputSize / sizeof(float));
    status = cudaMemcpy(outputHostData.data(), outputDevice, outputSize,
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            "Failed to copy data from device to host for output.");
    }
    std::cout << "Output data copied from device to host." << std::endl;
    std::cout << "First 10 output values: ";
    for (int i = 0; i < 10; ++i)
        std::cout << outputHostData[i] << " ";
    std::cout << std::endl;

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
