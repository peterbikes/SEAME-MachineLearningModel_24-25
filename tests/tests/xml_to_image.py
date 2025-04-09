import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os

def generate_lane_image(xml_path, output_image_path, img_size=(256, 256)):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Create a black image of the specified size
    lane_image = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    
    # Iterate through each image element in the XML
    for image_element in root.findall('image'):
        # Get all polygons (lane markings)
        for polygon in image_element.findall('polygon'):
            # Get points from the polygon
            points_str = polygon.get('points')
            points = [tuple(map(int, point.split(','))) for point in points_str.split(';')]
            points = np.array(points, dtype=np.int32)
            
            # Draw the polygon on the image
            cv2.fillPoly(lane_image, [points], color=255)  # 255 to mark the lane
            
    # Save the generated lane image
    cv2.imwrite(output_image_path, lane_image)
    print(f"Lane image saved to {output_image_path}")

def process_xmls_in_folder(xml_folder, output_folder):
    # Loop through all XML files in the folder
    for xml_name in os.listdir(xml_folder):
        if xml_name.endswith('.xml'):  # Process only XML files
            xml_path = os.path.join(xml_folder, xml_name)
            output_image_path = os.path.join(output_folder, f"{os.path.splitext(xml_name)[0]}.png")
            
            # Generate the lane image for the XML file
            generate_lane_image(xml_path, output_image_path)

if __name__ == "__main__":
    # Set the paths
    xml_folder = "tests"  # Folder containing XML files
    output_folder = xml_folder  # Save images in the same folder as XML files

    # Process all XML files in the folder
    process_xmls_in_folder(xml_folder, output_folder)

