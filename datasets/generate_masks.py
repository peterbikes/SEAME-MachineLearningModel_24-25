import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Path to CVAT annotation file and images
ANNOTATION_FILE = "annotations.xml"
IMAGES_FOLDER = "images"
MASKS_FOLDER = "masks"

os.makedirs(MASKS_FOLDER, exist_ok=True)

# Load the XML annotation file
tree = ET.parse(ANNOTATION_FILE)
root = tree.getroot()

for image in root.findall("image"):
    filename = image.get("name")
    width = int(image.get("width"))
    height = int(image.get("height"))
    
    mask = np.zeros((height, width), dtype=np.uint8)

    for polygon in image.findall("polygon"):
        label = polygon.get("label")
        points = polygon.get("points")
        points = np.array([list(map(float, pt.split(","))) for pt in points.split(";")], np.int32)

        # Assign high-intensity class IDs for visibility
        # white for lane, gray for path
        class_id = 255 if label == "lane" else 128  

        cv2.fillPoly(mask, [points], class_id)

    # Save the mask
    mask_filename = os.path.join(MASKS_FOLDER, filename.replace(".jpg", ".png"))
    cv2.imwrite(mask_filename, mask)

cv2.destroyAllWindows()
print("Mask generation complete!")

