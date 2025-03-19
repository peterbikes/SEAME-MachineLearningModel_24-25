import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# U-Net Model definition
def unet(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)
    
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    
    up6 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv4])
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    
    up7 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    
    up8 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    
    up9 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    return models.Model(inputs, outputs)

# Load images and masks
def load_dataset(dataset_dir, img_size=(256, 256)):
    image_dir = os.path.join(dataset_dir, "images")
    mask_dir = os.path.join(dataset_dir, "masks")
    images, masks = [], []
    
    for file in sorted(os.listdir(image_dir)):
        img = load_img(os.path.join(image_dir, file), target_size=img_size)
        mask = load_img(os.path.join(mask_dir, file.replace('.jpg', '.png')), target_size=img_size, color_mode='grayscale')
        images.append(img_to_array(img) / 255.0)
        masks.append(img_to_array(mask) / 255.0)
    
    return np.array(images), np.array(masks)

# Load all datasets
dataset_root = "datasets"
datasets = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
all_images, all_masks = [], []

for dataset in datasets:
    img, mask = load_dataset(os.path.join(dataset_root, dataset))
    all_images.extend(img)
    all_masks.extend(mask)

all_images = np.array(all_images)
all_masks = np.array(all_masks)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(all_images, all_masks, test_size=0.15, random_state=42)

# Load or create model
model_path = "unet_lane_detection_model.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    model = unet(input_size=(256, 256, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=4)

# Save the trained model
model.save(model_path)

# Evaluate and predict
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

output_folder = "latest_training_predictions"
os.makedirs(output_folder, exist_ok=True)

predictions = model.predict(X_val)
for i in range(len(predictions)):
    predicted_mask = (predictions[i].reshape(256, 256) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_folder, f"predicted_mask_{i}.png"), predicted_mask)

# Update trained datasets log
log_file = "trained_datasets.txt"
with open(log_file, "w") as f:
    for dataset in datasets:
        f.write(f"{dataset} ✔\n")

print(f"Predictions saved in {output_folder}")
print("Training complete. Trained datasets updated.")
