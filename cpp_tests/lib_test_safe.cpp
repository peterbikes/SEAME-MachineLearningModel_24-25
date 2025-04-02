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
const std::string model_path = "correct.trt";

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
nvinfer1::ICudaEngine* loadEngine(const std::string& trt_engine_path);
void allocateBuffers(nvinfer1::ICudaEngine* engine, std::vector<void*>& inputs, std::vector<void*>& outputs, std::vector<void*>& bindings, cudaStream_t& stream);
void processImagesInFolder(const std::string& folder, nvinfer1::IExecutionContext* context, std::vector<void*>& inputs, std::vector<void*>& outputs, std::vector<void*>& bindings, cudaStream_t& stream, std::map<std::string, std::pair<std::vector<int>, std::vector<int>>>& lanes);
void saveLanes(const std::map<std::string, std::pair<std::vector<int>, std::vector<int>>>& lanes);

int main() {
    // Load the model (TensorRT engine)
    std::ifstream file(model_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file");
    }
    std::cout << "File loaded." << std::endl;

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    // Create TensorRT runtime
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

    // Allocate buffers
    std::cout << "Allocating buffers" << std::endl;
    std::vector<void*> inputs, outputs, bindings;
    cudaStream_t stream;
    allocateBuffers(engine, inputs, outputs, bindings, stream);

    std::cout << "Processing images" << std::endl;
    // Process images and extract lane data
    std::map<std::string, std::pair<std::vector<int>, std::vector<int>>> lanes;
    processImagesInFolder(image_folder, context, inputs, outputs, bindings, stream, lanes);

    std::cout << "Creating folder" << std::endl;
    // Create output folder if it doesn't exist
    fs::path lanes_output_folder = fs::path(image_folder).parent_path() / "lanes";
    if (!fs::exists(lanes_output_folder)) {
        fs::create_directory(lanes_output_folder);
    }

    // Save lane data to text files
    saveLanes(lanes);

    std::cout << "Process completed successfully!" << std::endl;
    return 0;
}

// Implementations of functions

nvinfer1::ICudaEngine* loadEngine(const std::string& trt_engine_path) {
    // Implement loading TensorRT engine logic here (deserialize and return engine)
    return nullptr;  // Placeholder
}

void allocateBuffers(nvinfer1::ICudaEngine* engine, std::vector<void*>& inputs, std::vector<void*>& outputs, std::vector<void*>& bindings, cudaStream_t& stream) {
    // Example buffer allocation code
    int bindingCount = engine->getNbBindings();
    bindings.resize(bindingCount);
    
    for (int i = 0; i < bindingCount; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
        }
        size *= sizeof(float);  // Assuming float input/output
        cudaError_t err = cudaMalloc(&bindings[i], size);
        if (err != cudaSuccess) {
            std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
            return;
        }
    }

    // Allocate CUDA stream
    cudaStreamCreate(&stream);
}

void processImagesInFolder(const std::string& folder, nvinfer1::IExecutionContext* context, std::vector<void*>& inputs, std::vector<void*>& outputs, std::vector<void*>& bindings, cudaStream_t& stream, std::map<std::string, std::pair<std::vector<int>, std::vector<int>>>& lanes) {
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.is_regular_file()) {
            std::string image_path = entry.path().string();
            if (image_path.substr(image_path.find_last_of('.') + 1) == "jpg" || image_path.substr(image_path.find_last_of('.') + 1) == "png" || image_path.substr(image_path.find_last_of('.') + 1) == "jpeg") {
                // Load and process the image
                cv::Mat img = cv::imread(image_path);
                if (img.empty()) {
                    std::cerr << "Failed to read image: " << image_path << std::endl;
                    continue;
                }

                // Resize image to fit model input size (ensure to match your model's input size)
                cv::Mat resized_img;
                cv::resize(img, resized_img, cv::Size(224, 224));  // Example, update according to your model's input size

                // Convert image to float and normalize (if needed)
                resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);
		std::cout << "resizing images " << std::endl;

                // Copy image data to CUDA memory (use cudaMemcpy or appropriate method for your input buffer)
                // Placeholder: You must adjust this code to match the actual input format of your model.
                cudaError_t err = cudaMemcpy(inputs[0], resized_img.data, resized_img.total() * resized_img.elemSize(), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
                    continue;
                }

                // Run inference (check for errors)
		bool erro = context->executeV2(bindings.data());
		if (!erro) {
		    std::cerr << "Error during execution." << std::endl;
		    continue;
		}

                // Process output here (assuming the output format of your model)
                // For example, if you have lane points in the output, extract and store them in lanes.
                lanes[image_path] = { {}, {} };  // Placeholder, replace with actual output processing logic
            }
        }
    }
}

void saveLanes(const std::map<std::string, std::pair<std::vector<int>, std::vector<int>>>& lanes) {
    for (const auto& lane_data : lanes) {
        const std::string& image_name = lane_data.first;
        const std::pair<std::vector<int>, std::vector<int>>& lane_points = lane_data.second;

        std::string txt_filename = image_name.substr(0, image_name.find_last_of('.')) + ".txt";
        std::ofstream file(txt_filename);

        file << "Left Lane: ";
        for (const int& point : lane_points.first) {
            file << point << " ";
        }
        file << "\nRight Lane: ";
        for (const int& point : lane_points.second) {
            file << point << " ";
        }
        file << std::endl;

        std::cout << "Saved: " << txt_filename << std::endl;
    }
}

