from tensorflow.keras.models import load_model

# Load the model
model = load_model('unet_lane_detection_model.h5')

# Print a summary of the model
model.summary()

for layer in model.layers:
    print(f"Layer: {layer.name}, Type: {type(layer)}")

# Get weights of a specific layer
weights = model.layers[0].get_weights()
print("First layer weights:", weights)
