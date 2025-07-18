import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

def predict(image_input, model_path, input_shape):
    model = tf.keras.models.load_model(model_path)
    
    if isinstance(image_input, str): # If it's a file path
        img = image.load_img(image_input, target_size=(input_shape[0], input_shape[1]))
    else: # Assume it's a Streamlit UploadedFile object
        pil_img = Image.open(image_input).convert('RGB') # Ensure 3 channels
        img = pil_img.resize((input_shape[0], input_shape[1])) # Resize PIL image

    img_array = image.img_to_array(img) # Convert PIL Image to numpy array
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    # Normalization is handled by the model/training pipeline, so we remove it here.
    # img_array /= 255.0 
    
    prediction = model.predict(img_array)
    return prediction