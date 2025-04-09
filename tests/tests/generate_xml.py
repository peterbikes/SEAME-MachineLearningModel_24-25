import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def generate_lane_xml(image_path, model, output_xml_path):
    # Load and preprocess the fresh image
    image = load_img(image_path, target_size=(256, 256))
    image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Predict the lane mask
    predicted_mask = model.predict(image_array)
    predicted_mask = predicted_mask[0]  # Remove batch dimension

    # Apply thresholding to the predicted mask
    _, binary_mask = cv2.threshold(predicted_mask, 0.5, 1.0, cv2.THRESH_BINARY)
    binary_mask = (binary_mask * 255).astype(np.uint8)  # Convert to 8-bit image for contour extraction

    # Find contours of the predicted lane
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create the XML structure
    root = ET.Element("annotation")
    image_element = ET.SubElement(root, "image", name=os.path.basename(image_path), width="256", height="256")
    
    # Create a polygon for each detected contour
    for contour in contours:
        polygon = ET.SubElement(image_element, "polygon", label="lane", source="auto", occluded="0")
        
        points = []
        for point in contour:
            points.append(f"{point[0][0]},{point[0][1]}")
        
        polygon.set("points", ";".join(points))
        polygon.set("z_order", "0")
    
    # Write the XML to a file
    tree = ET.ElementTree(root)
    tree.write(output_xml_path)

    print(f"XML file saved to {output_xml_path}")


def process_images_in_folder(image_folder, model, output_folder):
    # Loop through all images in the folder
    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):  # Process only image files
            image_path = os.path.join(image_folder, image_name)
            output_xml_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}.xml")
            
            # Generate the XML file for the image
            generate_lane_xml(image_path, model, output_xml_path)


if __name__ == "__main__":
    # Set the paths
    image_folder = "tests"  # Folder containing test images
    model_path = "unet_lane_detection_model.h5"  # Path to your trained U-Net model
    output_folder = image_folder  # Save XML files in the same folder as images

    # Load the pre-trained U-Net model
    model = load_model(model_path)
    
    # Process all images in the folder
    process_images_in_folder(image_folder, model, output_folder)

