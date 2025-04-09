import os
import cv2
import numpy as np

def draw_lanes_on_images(image_folder, lanes_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_folder, image_name)
            lanes_path = os.path.join(lanes_folder, f"{os.path.splitext(image_name)[0]}.txt")
            output_path = os.path.join(output_folder, image_name)
            
            if not os.path.exists(lanes_path):
                print(f"No lane data for {image_name}, skipping...")
                continue
            
            # Load original image
            image = cv2.imread(image_path)
            original_height, original_width = image.shape[:2]
            
            # Load lane coordinates
            with open(lanes_path, 'r') as file:
                lines = file.readlines()
                left_lane = []
                right_lane = []
                try:
                    left_lane = eval(lines[0].split(': ')[1].strip()) if lines and len(lines) > 0 and ': ' in lines[0] else []
                    right_lane = eval(lines[1].split(': ')[1].strip()) if len(lines) > 1 and ': ' in lines[1] else []
                except Exception as e:
                    print(f"Error parsing {lanes_path}: {e}")
                    continue
            
            # If no lanes are found, skip saving the image
            if not left_lane and not right_lane:
                print(f"No lanes detected for {image_name}, skipping image creation...")
                continue
            
            # Scaling factors
            scale_x = original_width / 256
            scale_y = original_height / 256
            
            # Scale lane coordinates
            left_lane = [(int(x * scale_x), int(y * scale_y)) for x, y in left_lane] if left_lane else []
            right_lane = [(int(x * scale_x), int(y * scale_y)) for x, y in right_lane] if right_lane else []
            
            # Draw lanes on image
            if left_lane:
                cv2.polylines(image, [np.array(left_lane, np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)  # Red
            if right_lane:
                cv2.polylines(image, [np.array(right_lane, np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)  # Green
            
            # Save the output image
            cv2.imwrite(output_path, image)
            print(f"Processed {image_name} and saved to {output_path}")

if __name__ == "__main__":
    image_folder = "images"  # Folder with original images
    lanes_folder = "lanes"  # Folder with lane coordinate files
    output_folder = "output_images"  # Folder to save images with drawn lanes

    draw_lanes_on_images(image_folder, lanes_folder, output_folder)

