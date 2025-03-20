import os
import numpy as np
import cv2

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

def extract_lanes_from_mask(mask_path):
    """ Process a pre-generated mask instead of running a neural network. """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Unable to read mask {mask_path}")
        return None, None

    # Threshold mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if len(cnt) > 1]
    
    if len(contours) < 2:
        print(f"Warning: Detected less than two lanes in {mask_path}")
        return None, None
    
    # Sort contours based on x-coordinate
    contours = sorted(contours, key=lambda cnt: np.mean(cnt[:, 0, 0]))
    
    left_lane = contours[0]
    right_lane = contours[-1]
    
    left_centerline = compute_centerline(left_lane)
    right_centerline = compute_centerline(right_lane)

    return left_centerline, right_centerline

def process_masks_in_folder(mask_folder):
    results = {}
    for mask_name in os.listdir(mask_folder):
        if mask_name.endswith(('.jpg', '.png', '.jpeg')):
            mask_path = os.path.join(mask_folder, mask_name)
            left_lane, right_lane = extract_lanes_from_mask(mask_path)
            results[mask_name] = {"left_lane": left_lane, "right_lane": right_lane}
    return results

if __name__ == "__main__":
    mask_folder = "masks"  # Folder containing pre-generated segmentation masks
    lanes = process_masks_in_folder(mask_folder)

    lanes_output_folder = os.path.join(os.path.dirname(mask_folder), "lanes")
    os.makedirs(lanes_output_folder, exist_ok=True)

    for img, lane_data in lanes.items():
        txt_filename = os.path.splitext(img)[0] + ".txt"
        txt_filepath = os.path.join(lanes_output_folder, txt_filename)
        with open(txt_filepath, "w") as f:
            f.write(f"Left Lane: {lane_data['left_lane']}\n")
            f.write(f"Right Lane: {lane_data['right_lane']}\n")
        print(f"Saved: {txt_filepath}")

