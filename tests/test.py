import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load trained model
model = tf.keras.models.load_model('unet_lane_detection_model.h5')

# Load and preprocess image
image_path = 'test_image.jpg'
img_size = (256, 256)

image = load_img(image_path, target_size=img_size)
image_array = img_to_array(image) / 255.0  # Normalize
image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

# Predict mask
predicted_mask = model.predict(image_array)[0]  # Remove batch dimension

# Convert to uint8 format for saving
predicted_mask = (predicted_mask * 255).astype(np.uint8)

# Save the mask
cv2.imwrite('predicted_mask.png', predicted_mask)

print("Predicted mask saved as 'predicted_mask.png'")