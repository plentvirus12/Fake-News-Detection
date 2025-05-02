from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Enable CORS (optional, if frontend interacts)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = load_model('../models/hybrid_cnn_lstm_vit_model.h5')

# Preprocessing function
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image_bytes = await file.read()

        # Preprocess the image
        image = preprocess_image(image_bytes)

        # Prepare the second input (must match the required shape of (1, 768))
        second_input = np.zeros((image.shape[0], 768))  # Adjust to shape (batch_size, 768)

        # Prediction: pass both inputs (image and second_input)
        prediction = model.predict([image, second_input])

        # Process the prediction
        label = "Fake News" if prediction[0] > 0.5 else "Real News"
        confidence = float(prediction[0][0])

        return JSONResponse(content={
            "prediction": label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
