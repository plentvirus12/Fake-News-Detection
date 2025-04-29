# fake_news_detection_local.py

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from transformers import ViTImageProcessor, TFAutoModel
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set image size and dataset path (corrected your path here ✅)
IMAGE_SIZE = (224, 224)
base_path = r"C:\Users\gaura\Desktop\fake-news-detection\FakeNews_Images"  # <-- Correct full path

# ViT setup
vit_model = TFAutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def load_images_from_folder(folder_path, label):
    X_cnn, X_vit, y = [], [], []
    for img_file in tqdm(os.listdir(folder_path)):
        try:
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)

            # CNN input
            cnn_img = np.array(img) / 255.0
            X_cnn.append(cnn_img)

            # ViT input
            vit_inputs = vit_processor(images=img, return_tensors="tf")
            vit_output = vit_model(**vit_inputs).last_hidden_state[:, 0, :].numpy()[0]
            X_vit.append(vit_output)

            y.append(label)
        except Exception as e:
            print(f"Skipped {img_file}: {e}")
    return X_cnn, X_vit, y

def load_dataset(folder):
    X_cnn, X_vit, y = [], [], []
    for label, category in enumerate(["real", "fake"]):
        folder_path = os.path.join(base_path, folder, category)
        cnn_data, vit_data, labels = load_images_from_folder(folder_path, label)
        X_cnn += cnn_data
        X_vit += vit_data
        y += labels
    return np.array(X_cnn), np.array(X_vit), np.array(y)

# Load train, val, test datasets
X_cnn_train, X_vit_train, y_train = load_dataset("train")
X_cnn_val, X_vit_val, y_val = load_dataset("val")
X_cnn_test, X_vit_test, y_test = load_dataset("test")

def build_hybrid_model():
    # CNN + LSTM branch
    cnn_input = tf.keras.Input(shape=(224, 224, 3), name="cnn_input")
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(cnn_input)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Reshape((49, -1))(x)
    x = tf.keras.layers.LSTM(128)(x)

    # ViT branch
    vit_input = tf.keras.Input(shape=(768,), name="vit_input")

    # Merge both
    merged = tf.keras.layers.Concatenate()([x, vit_input])
    dense = tf.keras.layers.Dense(256, activation='relu')(merged)
    drop = tf.keras.layers.Dropout(0.3)(dense)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(drop)

    model = tf.keras.Model(inputs=[cnn_input, vit_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build and summarize model
model = build_hybrid_model()
model.summary()

# Training
history = model.fit(
    [X_cnn_train, X_vit_train], y_train,
    validation_data=([X_cnn_val, X_vit_val], y_val),
    epochs=10,
    batch_size=16
)

# Evaluate the Model
loss, accuracy = model.evaluate([X_cnn_test, X_vit_test], y_test)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# Save the model locally
os.makedirs(r"C:\Users\gaura\Desktop\fake-news-detection\models", exist_ok=True)  # Save path corrected
model.save(r"C:\Users\gaura\Desktop\fake-news-detection\models\hybrid_cnn_lstm_vit_model.h5")

# Predictions
y_pred_probs = model.predict([X_cnn_test, X_vit_test])
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = ['Real', 'Fake']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Hybrid Model (CNN + LSTM + ViT)")
plt.show()

# Classification Report
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=labels))

# CNN Only Model
def cnn_model():
    input_layer = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(input_layer)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_only = cnn_model()
cnn_only.fit(X_cnn_train, y_train, epochs=5, validation_data=(X_cnn_val, y_val))
cnn_acc = cnn_only.evaluate(X_cnn_test, y_test)[1]

# ViT Only Model
def vit_classifier():
    input_layer = tf.keras.Input(shape=(768,))
    x = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

vit_only = vit_classifier()
vit_only.fit(X_vit_train, y_train, epochs=5, validation_data=(X_vit_val, y_val))
vit_acc = vit_only.evaluate(X_vit_test, y_test)[1]

# Accuracy Comparison
print("\n✅ Model Accuracy Comparison:")
print(f"Hybrid Model (CNN + LSTM + ViT): {accuracy*100:.2f}%")
print(f"CNN Only: {cnn_acc*100:.2f}%")
print(f"ViT Only: {vit_acc*100:.2f}%")
