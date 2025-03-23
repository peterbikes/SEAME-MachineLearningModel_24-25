import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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

def extract_lanes(image_path, model):
    """ Use the trained model to predict lane masks and extract centerlines. """
    image = load_img(image_path, target_size=(256, 256))
    image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)

    predicted_mask = model.predict(image_array)[0]
    predicted_mask = (predicted_mask * 255).astype(np.uint8)
    
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

def process_images_in_folder(image_folder, model):
    results = {}
    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_folder, image_name)
            left_lane, right_lane = extract_lanes(image_path, model)
            results[image_name] = {"left_lane": left_lane, "right_lane": right_lane}
    return results

if __name__ == "__main__":
    image_folder = "images"
    model_path = "unet_lane_detection_model.h5"
    model = load_model(model_path)
    lanes = process_images_in_folder(image_folder, model)

    lanes_output_folder = os.path.join(os.path.dirname(image_folder), "lanes")
    os.makedirs(lanes_output_folder, exist_ok=True)

    for img, lane_data in lanes.items():
        txt_filename = os.path.splitext(img)[0] + ".txt"
        txt_filepath = os.path.join(lanes_output_folder, txt_filename)
        with open(txt_filepath, "w") as f:
            f.write(f"Left Lane: {lane_data['left_lane']}\n")
            f.write(f"Right Lane: {lane_data['right_lane']}\n")
        print(f"Saved: {txt_filepath}")

