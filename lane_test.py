import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def compute_centerline(contour):
    """ Compute the centerline by averaging x-coordinates for each unique y. """
    points = contour[:, 0, :]  # Extract (x, y) pairs
    unique_y = np.unique(points[:, 1])  # Get unique y-values
    centerline = []

    for y in unique_y:
        x_values = points[points[:, 1] == y, 0]  # Get all x-values at this y
        avg_x = int(np.mean(x_values))  # Compute the average x
        centerline.append((avg_x, int(y)))

    return centerline


def extract_lanes(image_path, model):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(256, 256))
    image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict the lane mask
    predicted_mask = model.predict(image_array)[0]

    # Apply thresholding
    _, binary_mask = cv2.threshold(predicted_mask, 0.5, 1.0, cv2.THRESH_BINARY)
    binary_mask = (binary_mask * 255).astype(np.uint8)

    # Find lane contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out single-point lanes
    contours = [cnt for cnt in contours if len(cnt) > 1]

    if len(contours) < 2:
        print(f"Warning: Detected less than two lanes in {image_path}")
        return None, None

    # Sort contours based on x-coordinate
    contours = sorted(contours, key=lambda cnt: np.mean(cnt[:, 0, 0]))

    # Select left and right lanes
    left_lane = contours[0]
    right_lane = contours[-1]

    # Compute centerlines
    left_centerline = compute_centerline(left_lane)
    right_centerline = compute_centerline(right_lane)

    return left_centerline, right_centerline


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

# Ensure the output directory exists
lanes_output_folder = os.path.join(os.path.dirname(image_folder), "lanes")
os.makedirs(lanes_output_folder, exist_ok=True)

for img, lane_data in lanes.items():
    txt_filename = os.path.splitext(img)[0] + ".txt"
    txt_filepath = os.path.join(lanes_output_folder, txt_filename)

    with open(txt_filepath, "w") as f:
        f.write(f"Left Lane: {lane_data['left_lane']}\n")
        f.write(f"Right Lane: {lane_data['right_lane']}\n")

    print(f"Saved: {txt_filepath}")

