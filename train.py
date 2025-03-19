import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configuration
DATASETS_DIR = "datasets"
TRAINED_DATASETS_FILE = os.path.join(DATASETS_DIR, "trained_datasets.txt")
MODEL_PATH = "unet_lane_detection_model.h5"
IMG_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 20

# Load trained datasets list
def get_trained_datasets():
    if not os.path.exists(TRAINED_DATASETS_FILE):
        return set()
    with open(TRAINED_DATASETS_FILE, "r") as f:
        return set(line.strip() for line in f)

# Save updated trained datasets
def update_trained_datasets(selected_datasets):
    trained = get_trained_datasets().union(selected_datasets)
    with open(TRAINED_DATASETS_FILE, "w") as f:
        f.writelines(f"{d}\n" for d in trained)

# Load images and masks
def load_images_and_masks(image_dir, mask_dir):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    images, masks = [], []
    for img_file in image_files:
        mask_file = img_file.replace(".jpg", ".png")
        if mask_file not in mask_files:
            continue  # Ensure corresponding mask exists
        
        image = load_img(os.path.join(image_dir, img_file), target_size=IMG_SIZE)
        mask = load_img(os.path.join(mask_dir, mask_file), target_size=IMG_SIZE, color_mode="grayscale")
        
        images.append(img_to_array(image) / 255.0)
        masks.append(img_to_array(mask) / 255.0)
    
    return np.array(images), np.array(masks)

# List available datasets
def list_datasets():
    trained_datasets = get_trained_datasets()
    dataset_folders = sorted([d for d in os.listdir(DATASETS_DIR) if os.path.isdir(os.path.join(DATASETS_DIR, d))])
    
    print("Available datasets:")
    for i, dataset in enumerate(dataset_folders, 1):
        mark = "✔" if dataset in trained_datasets else ""
        print(f"{i} - {dataset} {mark}")
    
    return dataset_folders

# Get user selection
def select_datasets(dataset_folders):
    while True:
        selection = input("Enter dataset numbers separated by space: ").strip()
        indices = selection.split()
        
        try:
            selected = [dataset_folders[int(i) - 1] for i in indices]
            return selected
        except (IndexError, ValueError):
            print("Invalid selection. Try again.")

# Main training function
def retrain_model():
    dataset_folders = list_datasets()
    selected_datasets = select_datasets(dataset_folders)
    
    all_images, all_masks = [], []
    for dataset in selected_datasets:
        img_dir = os.path.join(DATASETS_DIR, dataset, "images")
        mask_dir = os.path.join(DATASETS_DIR, dataset, "masks")
        
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            images, masks = load_images_and_masks(img_dir, mask_dir)
            all_images.extend(images)
            all_masks.extend(masks)
    
    if not all_images:
        print("No valid datasets selected.")
        return
    
    X_train, X_val, y_train, y_val = train_test_split(np.array(all_images), np.array(all_masks), test_size=0.15, random_state=42)
    model = load_model(MODEL_PATH)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    model.save(MODEL_PATH)
    update_trained_datasets(selected_datasets)
    print("Model retraining complete.")

if __name__ == "__main__":
    retrain_model()
