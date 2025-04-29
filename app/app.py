import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('path_to_your_model/hybrid_cnn_lstm_vit_model.h5')

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to the input size of your model
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Title and description
st.title("Fake News Detection App")
st.write("Upload an image to check if it's fake or real.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image and predict
    image = preprocess_image(image)
    prediction = model.predict(image)

    # Display result
    if prediction[0] > 0.5:
        st.write("Prediction: Fake News")
    else:
        st.write("Prediction: Real News")
