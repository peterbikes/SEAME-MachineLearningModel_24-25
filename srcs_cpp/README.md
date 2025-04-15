compile and run :

clear && sudo c++ -std=c++17 -o test_libraries lib_test.cpp -fopenmp `pkg-config --cflags --libs opencv4` -lnvinfer -lcudart -I/usr/include/eigen3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lstdc++fs && sudo ./test_libraries

# steps taken so far:

- engine is read and deserialized directly on main;
- an execution context is created with the engine;
- memory is allocated according to model's input and output:

  - input = 256 * 256 * 3 (image size * RGB) ;
  - output = 256 * 256 * 1 (lane or no lane, black or white)
- image is loaded, resized and normalized using loadImage(). This function returns a vector std::vector `<float>`og_image(256 * 256 * 3) , that corresponds to the original loaded image, treated. this image is then saved in as normalized_image in order to be analyzed - the process is done correctly;
- this vector is then copied from CPU to GPU using cudaMemcpy
- 2 void pointers are created for input (data that we will treat) and output (for retrieving the result) . One holds the image, the other is empty for now.
- inference is run using context->executeV2
- cudaMemcpy is used to copy the data in ouputHostData to the vector outputDevice, from GPU to CPU.
- The data is then converted to a mask using openCV.

  There is another function, voidcheckEngineSpecs(nvinfer1::ICudaEngine*engine), that analyses the engine, checks inputs and outputs and it's dimensions

  # Possible issues:
- I might be treating the images incorrectly, but our python code seems to do the same, you can check it at:
- Inference might be "failing" and it is really hard to check,
- Cuda memory allocation;
- pointers?
