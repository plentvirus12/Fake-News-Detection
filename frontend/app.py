import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Define the FastAPI server URL
API_URL = "http://localhost:8000/predict/"  

# Set up the Streamlit app title
st.title("📰 Fake News Detection Using Hybrid CNN-LSTM-ViT Model")

# Upload image file
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded file to bytes
    img_bytes = uploaded_file.read()

    # Send the image to the FastAPI backend
    try:
        # Include filename and content type
        response = requests.post(
            API_URL,
            files={"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
        )

        if response.status_code == 200:
            prediction = response.json()
            label = prediction["prediction"]
            confidence = prediction["confidence"]

            # Display prediction result
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {confidence * 100:.2f}%")
        else:
            st.error(f"Error in prediction: {response.status_code}")
            st.text(f"Response: {response.text}")

    except Exception as e:
        st.error(f"Error occurred: {e}")
