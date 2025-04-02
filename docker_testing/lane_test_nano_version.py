import os
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(trt_engine_path):
    """Load TensorRT engine."""
    with open(trt_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    """Allocate GPU buffers for inference."""
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        buffer = cuda.mem_alloc(size * dtype().itemsize)
        bindings.append(int(buffer))
        if engine.binding_is_input(binding):
            inputs.append(buffer)
        else:
            outputs.append(buffer)
    return inputs, outputs, bindings, stream

def infer(context, inputs, outputs, bindings, stream, image_array):
    """Run inference with TensorRT."""
    cuda.memcpy_htod_async(inputs[0], image_array, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0], outputs[0], stream)
    stream.synchronize()
    return outputs[0]

def preprocess_image(image_path):
    """Load and preprocess the image."""
    image = Image.open(image_path).resize((256, 256))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dim
    return np.ascontiguousarray(image_array)

def rdp_simplify(points, epsilon=5.0):
    """ Apply the Ramer-Douglas-Peucker algorithm to simplify a curve. """
    points = np.array(points, dtype=np.int32)
    return cv2.approxPolyDP(points, epsilon, False)[:, 0, :].tolist()

def compute_centerline(contour):
    """ Compute the centerline by averaging x-coordinates for each unique y. """
    points = contour[:, 0, :]
    unique_y = np.unique(points[:, 1])
    centerline = []

    for y in unique_y:
        x_values = points[points[:, 1] == y, 0]
        avg_x = int(np.mean(x_values))
        centerline.append((avg_x, int(y)))

    return rdp_simplify(centerline)

def extract_lanes(image_path, context, inputs, outputs, bindings, stream):
    """ Run TensorRT model to detect lane masks and extract centerlines. """
    image_array = preprocess_image(image_path)
    output = infer(context, inputs, outputs, bindings, stream, image_array)
    
    predicted_mask = (output * 255).astype(np.uint8).reshape((256, 256))
    _, binary_mask = cv2.threshold(predicted_mask, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if len(cnt) > 1]
    
    if len(contours) == 0:
        print(f"Warning: No lanes detected in {image_path}")
        return None, None
    
    contours = sorted(contours, key=lambda cnt: np.mean(cnt[:, 0, 0]))
    
    left_lane = compute_centerline(contours[0]) if len(contours) > 0 else None
    right_lane = compute_centerline(contours[-1]) if len(contours) > 1 else None
    
    return left_lane, right_lane

def process_images_in_folder(image_folder, context, inputs, outputs, bindings, stream):
    results = {}
    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_folder, image_name)
            left_lane, right_lane = extract_lanes(image_path, context, inputs, outputs, bindings, stream)
            results[image_name] = {"left_lane": left_lane, "right_lane": right_lane}
    return results

if __name__ == "__main__":
    image_folder = "images"
    model_path = "model.onnx"

    engine = load_engine(model_path)
    if engine is None:
        print("Engine failed to load.")
    else:
        context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    lanes = process_images_in_folder(image_folder, context, inputs, outputs, bindings, stream)

    lanes_output_folder = os.path.join(os.path.dirname(image_folder), "lanes")
    os.makedirs(lanes_output_folder, exist_ok=True)

    for img, lane_data in lanes.items():
        txt_filename = os.path.splitext(img)[0] + ".txt"
        txt_filepath = os.path.join(lanes_output_folder, txt_filename)
        with open(txt_filepath, "w") as f:
            f.write(f"Left Lane: {lane_data['left_lane']}\n")
            f.write(f"Right Lane: {lane_data['right_lane']}\n")
        print(f"Saved: {txt_filepath}")

