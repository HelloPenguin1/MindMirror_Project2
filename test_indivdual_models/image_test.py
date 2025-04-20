import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import json
import cv2
from tkinter import Tk, filedialog, Label  # Ensure Tkinter is imported
import tkinter as tk  # Add this line to alias tkinter as tk
from PIL import Image, ImageTk

# Load configuration
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found. Please ensure it exists.")
    exit(1)

# Load emotion mapping
try:
    with open('emotion_map.json', 'r') as f:
        emotion_map = json.load(f)
except FileNotFoundError:
    print("Error: emotion_map.json not found. Using default mapping.")
    emotion_map = {0: 'surprise', 1: 'fear', 2: 'disgust', 3: 'happiness', 4: 'sadness', 5: 'anger', 6: 'neutrality'}

# Preprocessing function
def convert_to_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img

def preprocess_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = cv2.resize(img, tuple(config['image_size']))
    img_array = np.expand_dims(img, axis=0)
    img_array = convert_to_rgb(img_array[0])
    return np.expand_dims(img_array, axis=0)

# Load the image model
try:
    image_model = load_model(config['model_paths']['image'])
except Exception as e:
    print(f"Error loading image model: {e}")
    exit(1)

# GUI for image upload and prediction
def predict_emotion():
    root = Tk()
    root.title("Image Emotion Predictor")
    root.geometry("400x300")

    # Label to display result
    result_label = Label(root, text="Upload a JPG image to predict emotion", wraplength=350)
    result_label.pack(pady=20)

    # Function to handle file upload and prediction
    def upload_image():
        file_path = filedialog.askopenfilename(filetypes=[("JPG files", "*.jpg")])
        if file_path:
            # Load and preprocess image
            img = cv2.imread(file_path)
            if img is None:
                result_label.config(text="Error: Could not load image.")
                return
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_data = preprocess_image(img_rgb)

            # Predict emotion
            prediction = image_model.predict(img_data, verbose=0)
            emotion_idx = np.argmax(prediction, axis=1)[0]
            confidence = float(np.max(prediction)) * 100
            emotion = emotion_map.get(str(emotion_idx), 'unknown')

            # Display result
            result_label.config(text=f"Predicted emotion: {emotion}\nConfidence: {confidence:.2f}%")

    # Upload button
    upload_button = tk.Button(root, text="Upload Image", command=upload_image)
    upload_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    predict_emotion()