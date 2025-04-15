import tensorflow as tf
import tf2onnx
import onnx
from create_model import unet

def convert_keras_to_onnx(keras_model_path, onnx_model_path):
    """
    Convert a Keras model (.keras file) to ONNX format using tf2onnx.
    
    Parameters:
    keras_model_path (str): Path to the input Keras model file
    onnx_model_path (str): Path where the ONNX model will be saved
    """
    try:
        # Load the Keras model
        keras_model = tf.keras.models.load_model(keras_model_path)
        
        # Convert to ONNX using tf2onnx
        input_signature = [tf.TensorSpec(shape=keras_model.inputs[0].shape, 
                                         dtype=keras_model.inputs[0].dtype)]
        
        # Convert the model
        onnx_model, _ = tf2onnx.convert.from_keras(keras_model, 
                                                   input_signature, 
                                                   opset=13)
        
        # Save the ONNX model
        onnx.save(onnx_model, onnx_model_path)
        
        print(f"Model successfully converted and saved to {onnx_model_path}")
    
    except Exception as e:
        print(f"Conversion error: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual model paths
    input_keras_model = "unet_lane_detection_model.h5"
    output_onnx_model = "model2.onnx"
    
    convert_keras_to_onnx(input_keras_model, output_onnx_model)
