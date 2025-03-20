# U-Net - Machine Learning Model Approach

## Results so far

Accuracy of around 80% with 20 Epochs based on 92 images/masks (vid1 folder only);

## Steps taken:

### 1- Mask generation

- Export as CVAT for images 1.1 from cvat.ai after treating the images;
- NOTE: as some images are trash, we later need to remove them in training phase (it's easier this way)
- run generate_masks script, having the annotations file exported sitting next to the script and the images folder

### 2- Training (train.py file)

**U-Net model definition**

* **Encoder (Downsampling):** Extracts features using convolution and max pooling.
* **Bottleneck:** Deepest layer capturing high-level representations.
* **Decoder (Upsampling):** Restores spatial resolution using transposed convolutions and skip connections.
* **Output layer:** Uses a **sigmoid activation** for binary segmentation (mask prediction). (?))
* **Model summary:** Displays architecture details.

- THIS CAN BE tweaked TO ACHIEVE HIGHER RESULTS

before running:

pip install "numpy<2" --force-reinstall

pip install scikit-learn

Training part of the code:

- loads images and masks, resizes them, normalizes them (?));
- The `train_test_split()` function splits the data into training and validation sets (80% training, 20% validation).

### 3 - Output:

lane_detection_model.h5 explanation

CHAT GPT warning:

"The file **`lane_detection_model.h5`** is a saved model file in the **HDF5** format, which is commonly used with **Keras** and  **TensorFlow** . It contains the architecture, weights, and training configuration of a machine learning model, which in your case, is likely designed for **lane detection** using computer vision and deep learning techniques (such as CNNs).

The file is typically generated after training a model, and it allows you to:

1. **Load the trained model** without needing to retrain it.
2. **Evaluate or make predictions** using the model on new data (like images or video frames from the JetRacer's camera).
3. **Reuse the model** for inference in different environments, such as deploying on the Raspberry Pi or other hardware.

In summary, **`lane_detection_model.h5`** holds the trained neural network that is used to detect lanes from camera input, and you can load it for inference or further tuning."

## File tree explanation

* **generate_masks.py** : Generates mask images based on input images and annotations file from cvat for training the model (converts images to appropriate mask format).
* **images** : Folder containing the raw images used for training and generating masks.
* **train.py** : Script for training the U-Net model on the image and mask data.

## Why U-Net?

### 1.  **Efficient Semantic Segmentation** :

* **U-Net** is specifically designed for semantic segmentation tasks, which makes it highly effective for pixel-wise classification. In lane detection, this means U-Net can accurately classify each pixel as part of the lane or not, which is essential for detecting lanes from images. Its ability to segment road areas, including complex lanes in different lighting conditions, is a major advantage.

### 2.  **Lightweight and Optimized for Embedded Systems** :

* The **Jetson Nano** is a resource-constrained device, but U-Net can be adapted to work efficiently even on such low-power hardware. The model can be optimized for performance using quantization or pruning techniques, ensuring real-time lane detection without heavy computational overhead, which is crucial for embedded systems like the Jetson Nano.

### 3.  **ROS Integration** :

* U-Net can easily be integrated into **ROS** (Robot Operating System), which is widely used in robotics and autonomous vehicle systems. ROS supports modular software development, making it simple to connect the lane detection model with other systems (e.g., control systems, sensors, or planning algorithms) via ROS nodes. U-Net’s output can be seamlessly passed as input for other processes like steering or speed control in autonomous vehicles.

### 4.  **Robust to Variations in Road Conditions** :

* U-Net's architecture, with its use of skip connections, enables it to capture both high-level context and fine-grained details, making it highly robust in various conditions. This capability is useful for lane detection as it can handle complex scenes with curves, intersections, varying road textures, or partial occlusions (e.g., from other vehicles). It can effectively detect lanes in dynamic environments like highways or city streets.

### 5.  **Adaptable to Custom Datasets** :

* U-Net is highly customizable and can be fine-tuned on specific datasets. This allows you to train the model on your own lane detection dataset, improving performance for specific road conditions, weather, or geographical regions. The flexibility to adapt the model to custom data makes it suitable for deployment in real-world lane detection tasks, ensuring high accuracy in different environments.

## NEXT STEPS:

- [X] Make sure the masks used are generated using cvat.ai
- [X] (???)  RESERVE AROUND 10% of collected images for control - training model does this already with 20%

- Feed the model more images;
- Improve model definition
- Improve overall code because it is a bit rubbish
