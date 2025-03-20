import os
import cv2
import numpy as np

def draw_lanes_on_images(image_folder, lanes_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for image_name in os.listdir(image_folder):
        if not image_name.endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        image_path = os.path.join(image_folder, image_name)
        txt_path = os.path.join(lanes_folder, os.path.splitext(image_name)[0] + '.txt')
        output_path = os.path.join(output_folder, image_name)
        
        if not os.path.exists(txt_path):
            print(f"No lane data for {image_name}, skipping.")
            continue
        
        # Read lane coordinates from the text file
        with open(txt_path, 'r') as file:
            lines = file.readlines()
        
        left_lane = eval(lines[0].split(':')[1].strip()) if 'Left Lane' in lines[0] else None
        right_lane = eval(lines[1].split(':')[1].strip()) if len(lines) > 1 and 'Right Lane' in lines[1] else None
        
        if not left_lane and not right_lane:
            print(f"No valid lane data in {txt_path}, skipping.")
            continue
        
        # Load image
        image = cv2.imread(image_path)
        
        # Draw left lane in blue if available
        if left_lane:
            left_lane = np.array(left_lane, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [left_lane], isClosed=False, color=(255, 0, 0), thickness=2)
        
        # Draw right lane in red if available
        if right_lane:
            right_lane = np.array(right_lane, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [right_lane], isClosed=False, color=(0, 0, 255), thickness=2)
        
        # Save output image
        cv2.imwrite(output_path, image)
        print(f"Processed {image_name} and saved to {output_path}")

# Define folders
image_folder = "images"
lanes_folder = "lanes"
output_folder = "output_images"

# Run the function
draw_lanes_on_images(image_folder, lanes_folder, output_folder)

