import os
import cv2
import ast
import numpy as np

def draw_lanes_on_images(image_folder, lanes_folder, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through all images in the image folder
    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_folder, image_name)
            txt_path = os.path.join(lanes_folder, f"{os.path.splitext(image_name)[0]}.txt")
            output_path = os.path.join(output_folder, image_name)
            
            # Check if corresponding text file exists
            if not os.path.exists(txt_path):
                print(f"Skipping {image_name}: No matching lane file found.")
                continue
            
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image {image_name}")
                continue
            
            # Read lane coordinates from the text file
            with open(txt_path, 'r') as file:
                lines = file.readlines()
                left_lane = ast.literal_eval(lines[0].split(': ')[1].strip())
                right_lane = ast.literal_eval(lines[1].split(': ')[1].strip())
            
            # Convert points to numpy arrays
            left_lane = np.array(left_lane, np.int32).reshape((-1, 1, 2))
            right_lane = np.array(right_lane, np.int32).reshape((-1, 1, 2))
            
            # Draw lanes on the image
            cv2.polylines(image, [left_lane], isClosed=False, color=(255, 0, 0), thickness=2)  # Blue for left lane
            cv2.polylines(image, [right_lane], isClosed=False, color=(0, 0, 255), thickness=2)  # Red for right lane
            
            # Save the processed image
            cv2.imwrite(output_path, image)
            print(f"Processed {image_name} and saved to {output_path}")

if __name__ == "__main__":
    # Define directories
    image_folder = "images"
    lanes_folder = "lanes"
    output_folder = "output_images"
    
    # Run lane drawing process
    draw_lanes_on_images(image_folder, lanes_folder, output_folder)

