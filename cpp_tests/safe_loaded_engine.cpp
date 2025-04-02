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
        // Print all messages, including VERBOSE
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

// Function declarations (you would need to implement them based on your original code)
nvinfer1::ICudaEngine* loadEngine(const std::string& trt_engine_path);
void allocateBuffers(nvinfer1::ICudaEngine* engine, std::vector<void*>& inputs, std::vector<void*>& outputs, std::vector<void*>& bindings, cudaStream_t& stream);
void processImagesInFolder(const std::string& folder, nvinfer1::IExecutionContext* context, std::vector<void*>& inputs, std::vector<void*>& outputs, std::vector<void*>& bindings, cudaStream_t& stream, std::map<std::string, std::pair<std::vector<int>, std::vector<int>>>& lanes);
void saveLanes(const std::map<std::string, std::pair<std::vector<int>, std::vector<int>>>& lanes);

int main() {

	std::ifstream file(model_path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to open engine file");
    }
    std::cout << "file loaded (?)" << std::endl;

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
//    context.reset(engine->createExecutionContext());

	std::cout << "end" << std::endl;

//	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
//	if (!runtime) {
//	    std::cerr << "Error: Failed to create TensorRT runtime" << std::endl;
//	    return -1;
//	} else {
//		std::cout << "TensorRT runtime created." << std::endl;
//	}
//
//
//    // Load the TensorRT engine
//    std::ifstream file(model_path);
//	if (!file.good()) {
//    		std::cerr << "Error: Model file not found at " << model_path << std::endl;
//    		return -1;
//	}
//    std::cout << "Model path: " << model_path << std::endl;
//    nvinfer1::ICudaEngine* engine = loadEngine(model_path);
//    std::cout << "Engine: " << engine << std::endl;
//    std::cout << "Model absolute path: " << std::filesystem::absolute(model_path) << std::endl;
//    if (engine == nullptr) {
//        std::cerr << "Engine failed to load." << std::endl;
//        return -1;
//    }
//
//    // Create execution context
//    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
//    if (!context) {
//        std::cerr << "Failed to create execution context." << std::endl;
//        return -1;
//    }
//
//    // Allocate buffers
//    std::vector<void*> inputs, outputs, bindings;
//    cudaStream_t stream;
//    allocateBuffers(engine, inputs, outputs, bindings, stream);
//
//    // Process images and get lane data
//    std::map<std::string, std::pair<std::vector<int>, std::vector<int>>> lanes;
//    processImagesInFolder(image_folder, context, inputs, outputs, bindings, stream, lanes);
//
//    // Create output folder if it doesn't exist
//    fs::path lanes_output_folder = fs::path(image_folder).parent_path() / "lanes";
//    if (!fs::exists(lanes_output_folder)) {
//        fs::create_directory(lanes_output_folder);
//    }
//
//    // Save lane data to text files
//    saveLanes(lanes);
//
//    std::cout << "Process completed successfully!" << std::endl;
//    return 0;
}

// Implementations of functions (placeholders)
nvinfer1::ICudaEngine* loadEngine(const std::string& trt_engine_path) {
    // Implement loading TensorRT engine logic here (deserialize and return engine)
    return nullptr;  // Placeholder
}

void allocateBuffers(nvinfer1::ICudaEngine* engine, std::vector<void*>& inputs, std::vector<void*>& outputs, std::vector<void*>& bindings, cudaStream_t& stream) {
    // Implement buffer allocation logic for inputs, outputs, bindings, and stream here
}

void processImagesInFolder(const std::string& folder, nvinfer1::IExecutionContext* context, std::vector<void*>& inputs, std::vector<void*>& outputs, std::vector<void*>& bindings, cudaStream_t& stream, std::map<std::string, std::pair<std::vector<int>, std::vector<int>>>& lanes) {
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.is_regular_file()) {
            std::string image_path = entry.path().string();
            if (image_path.substr(image_path.find_last_of('.') + 1) == "jpg" || image_path.substr(image_path.find_last_of('.') + 1) == "png" || image_path.substr(image_path.find_last_of('.') + 1) == "jpeg") {
                // Call your lane extraction logic here and add to the lanes map
                lanes[image_path] = { {}, {} };  // Placeholder
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

