import gradio as gr
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('ethnicity_model.h5')

# Define a function to make predictions on images
def predict_ethnicity(input_image):
    # Convert Gradio image object to PIL image object
    pil_image = Image.fromarray(input_image.astype('uint8'), 'RGB')
    
    # Convert the image to grayscale
    pil_image_gray = pil_image.convert('L')
    
    # Resize the image
    pil_image_gray = pil_image_gray.resize((48, 48))
    
    # Convert PIL image to numpy array
    img_array = np.array(pil_image_gray)
    
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=2)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Decode prediction
    classes = ["white", "Black", "Asian", "Indian", "other"]
    max_index = np.argmax(prediction)
    max_percentage = float(prediction[0][max_index]) * 100
    max_ethnicity = classes[max_index]
    
    # Format output as text
    output_str = f"{max_ethnicity}: {max_percentage:.2f}%"
    return output_str

# Create a Gradio interface
iface = gr.Interface(
    fn=predict_ethnicity, 
    inputs="image", 
    outputs="text",  # Output as text
    title="Ethnicity and Race Classification",
    description="Upload an image to predict the ethnicity of the person."
)

# Launch the interface
iface.launch()
