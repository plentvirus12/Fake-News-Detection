from tensorflow.keras.models import load_model

# Load the model
model = load_model('hybrid_cnn_lstm_vit_model.h5')

# Print model summary
model.summary()
