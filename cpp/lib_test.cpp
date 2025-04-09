#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

const std::string image_folder = "images";
const std::string model_path = "correct.engine";

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        std::cerr << "[TensorRT] ";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: std::cerr << "[ERROR] "; break;
            case Severity::kERROR: std::cerr << "[ERROR] "; break;
            case Severity::kWARNING: std::cerr << "[WARNING] "; break;
            case Severity::kINFO: std::cerr << "[INFO] "; break;
            case Severity::kVERBOSE: std::cerr << "[VERBOSE] "; break;
            default: break;
        }
        std::cerr << msg << std::endl;
    }
} gLogger;

// Function declarations

int main() {
    // Load the model (TensorRT engine)
    std::ifstream file(model_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file");
    }
    std::cout << "File loaded." << std::endl;

    // seek the end of the file to determine engine size
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    // create a file to hold engine data
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    // Create TensorRT runtime - to deserealize model and process inference
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Error: Failed to create TensorRT runtime" << std::endl;
        return -1;
    }

    // Deserialize engine
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
    if (!engine) {
        std::cerr << "Error: Failed to deserialize engine" << std::endl;
        return -1;
    }

    std::cout << "Engine loaded successfully." << std::endl;

    // Create execution context
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Error: Failed to create execution context" << std::endl;
        return -1;
    }

	//create folder for results and things like that
    fs::create_directory("results");

    // path to the fucking image
    std::string image_path = "images/v1_frame_0186.jpg";
    cv::imwrite("results/original_pic.jpg", cv::imread(image_path));

	// Load the image
	cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);  // Load color image
	if (img.empty()) {
	    throw std::runtime_error("Failed to load image");
	}
	
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
	// Convert the image to grayscale (if needed)
//	cv::Mat gray_img;
//	if (img.channels() == 3) {
//	    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
//	} else {
//	    gray_img = img;  // Already grayscale
//	}
	
	// Resize the image to 28x28
	cv::Mat resized_img;
//	cv::resize(gray_img, resized_img, cv::Size(256, 256));
	cv::resize(img, resized_img, cv::Size(256, 256));
	
	// Normalize the image
	// resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);  // Normalize to [0, 1]
	
	// Optionally, display image details
	std::cout << "Resized image size: " << resized_img.size() << std::endl;
	cv::imwrite("results/resized_image.jpg", resized_img);


	// Flatten the 256x256 image into a 1D array
	std::vector<float> input_data(resized_img.total());
	for (int i = 0; i < resized_img.rows; ++i) {
	    for (int j = 0; j < resized_img.cols; ++j) {
	        input_data[i * resized_img.cols + j] = resized_img.at<float>(i, j);
	    }
	}
	
	// print the array
	//for (const auto& value : input_data) {
	//    std::cout << value << ' ';
	//}
	//std::cout << std::endl;
	

	// allocating memory - lost me on this pooint
	//std::vector<float> output_data(10, 0.0f);
	std::vector<float> output_data(1, 0.0f); // - > we only have one output (lane or no lane)

	// RUNNING INFERENCE
	// Create a stream for asynchronous execution
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	
	// Create GPU buffers for input and output
	void* d_input;
	void* d_output;
	cudaMalloc(&d_input, input_data.size() * sizeof(float));
	cudaMalloc(&d_output, output_data.size() * sizeof(float));
	
	// Copy input data to GPU
	cudaMemcpyAsync(d_input, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
	
	// Set up bindings
	void* bindings[] = {d_input, d_output};
	
	// Execute the inference
	context->enqueue(1, bindings, stream, nullptr);
	
	// Copy output data back to CPU
	cudaMemcpyAsync(output_data.data(), d_output, output_data.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
	
	// Wait for the stream to finish
	cudaStreamSynchronize(stream);
	
	// Clean up
	cudaStreamDestroy(stream);
	cudaFree(d_input);
	cudaFree(d_output);

	// Find the index of the maximum value in the output vector
auto max_iter = std::max_element(output_data.begin(), output_data.end());
int predicted_class = std::distance(output_data.begin(), max_iter);


// Define the threshold for mask creation
float threshold = 0.5f;

// Create a mask based on the threshold
cv::Mat mask(256, 256, CV_8UC1); // Single-channel mask
for (int i = 0; i < 256; ++i) {
    for (int j = 0; j < 256; ++j) {
        int index = i * 256 + j;
        mask.at<uchar>(i, j) = (output_data[index] > threshold) ? 255 : 0;
    }
}

// Resize the mask to match the original image size
cv::Mat mask_resized;
cv::resize(mask, mask_resized, cv::Size(1920, 1080), 0, 0, cv::INTER_LINEAR);

// Save the mask as an image
// cv::imwrite("results/mask.png", mask_resized);
cv::imwrite("results/mask_1920*1080-normalized-one_output-3.png", mask_resized);

    std::cout << "Process completed successfully!" << std::endl;
    return 0;
}
