import onnxruntime as ort
import numpy as np
import cv2
import os
import argparse

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize
    img_resized = cv2.resize(img, target_size)
    
    # Normalize (same as in your training)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Reshape to batch format [1, H, W, C]
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img, img_resized, img_batch

def run_inference(onnx_model_path, image_path):
    # Create output directory
    os.makedirs("onnx_test_results", exist_ok=True)
    
    # Load and preprocess image
    original_img, resized_img, input_data = load_and_preprocess_image(image_path)
    
    # Save preprocessed image
    cv2.imwrite("onnx_test_results/original.jpg", original_img)
    cv2.imwrite("onnx_test_results/resized.jpg", resized_img)
    
    # Create ONNX Runtime session
    print(f"Running inference with ONNX model: {onnx_model_path}")
    session = ort.InferenceSession(onnx_model_path)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    print(f"Input name: {input_name}")
    print(f"Input shape: {session.get_inputs()[0].shape}")
    
    # Run inference
    outputs = session.run(None, {input_name: input_data})
    output = outputs[0]
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: {output.min()} to {output.max()}")
    
    # Process and save output in different ways
    output_image = output[0].reshape(256, 256)
    
    # 1. Direct output (normalized to 0-255)
    normalized_output = ((output_image - output_image.min()) / 
                         (output_image.max() - output_image.min()) * 255).astype(np.uint8)
    cv2.imwrite("onnx_test_results/normalized_output.jpg", normalized_output)
    
    # 2. Inverted output
    inverted_output = 255 - normalized_output
    cv2.imwrite("onnx_test_results/inverted_output.jpg", inverted_output)
    
    # 3. Thresholded output (try different thresholds)
    for threshold in [0.1, 0.2, 0.3]:
        threshold_value = threshold
        thresholded = (output_image > threshold_value).astype(np.uint8) * 255
        cv2.imwrite(f"onnx_test_results/thresholded_{threshold}.jpg", thresholded)
    
    # 4. Apply sigmoid (in case it was lost in conversion)
    sigmoid_output = 1 / (1 + np.exp(-output_image))
    sigmoid_normalized = (sigmoid_output * 255).astype(np.uint8)
    cv2.imwrite("onnx_test_results/sigmoid_output.jpg", sigmoid_normalized)
    
    # 5. Create heatmap visualization
    heatmap = cv2.applyColorMap(normalized_output, cv2.COLORMAP_JET)
    cv2.imwrite("onnx_test_results/heatmap.jpg", heatmap)
    
    # 6. Overlay on original image
    overlay = cv2.addWeighted(
        resized_img, 0.7, 
        cv2.cvtColor(normalized_output, cv2.COLOR_GRAY2BGR), 0.3, 
        0
    )
    cv2.imwrite("onnx_test_results/overlay.jpg", overlay)
    
    print(f"Results saved to onnx_test_results directory")
    return output_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ONNX model inference")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--image", required=True, help="Path to input image file")
    
    args = parser.parse_args()
    run_inference(args.model, args.image)
