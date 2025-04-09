import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load trained model
model = tf.keras.models.load_model('../unet_lane_detection_model.h5')

# Load and preprocess image
image_path = 'images/v2_frame_0072.jpg'
img_size = (256, 256)

image = load_img(image_path, target_size=img_size)
image_array = img_to_array(image) / 255.0  # Normalize
image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

# Predict mask
predicted_mask = model.predict(image_array)[0]  # Remove batch dimension

# Convert to uint8 format for processing
predicted_mask = (predicted_mask * 255).astype(np.uint8)

# Apply morphological operations to remove noise
kernel = np.ones((3, 3), np.uint8)
predicted_mask = cv2.morphologyEx(predicted_mask, cv2.MORPH_OPEN, kernel)

# Find contours
contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

lanes = []
min_contour_length = 5  # Ignore very small contours

for contour in contours:
    lane = [(point[0][0], point[0][1]) for point in contour]
    if len(lane) > 1:  # Ignore single-point lanes
        lanes.append(lane)

# Output lane arrays
print("Number of lanes detected:", len(lanes))
for lane_number, lane in enumerate(lanes, start=1):
    print(f"Lane {lane_number} points:")
    print(lane)

# Save the mask
cv2.imwrite('predicted_mask.png', predicted_mask)

print("Predicted mask saved as 'result2.png'")

