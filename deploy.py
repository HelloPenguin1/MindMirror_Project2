import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import json
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import sys
from typing import Tuple, Dict, Optional

# Load configuration
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found. Please ensure it exists.")
    sys.exit(1)

# Load emotion mapping
try:
    with open('emotion_map.json', 'r') as f:
        emotion_map = json.load(f)
except FileNotFoundError:
    print("Error: emotion_map.json not found. Using default mapping.")
    emotion_map = {0: 'surprise', 1: 'fear', 2: 'disgust', 3: 'happiness', 4: 'sadness', 5: 'anger', 6: 'neutrality'}

# Load tokenizer
try:
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
except FileNotFoundError:
    print("Error: tokenizer.json not found. Please ensure it exists.")
    sys.exit(1)

# Personality mapping
PERSONALITY_MAPPING = {
    'happiness': {'Joy/Positive': {'Extraversion': 0.8, 'Agreeableness': 0.6}},
    'surprise': {'Admiration': {'Openness': 0.7, 'Extraversion': 0.5}},
    'fear': {'Sadness/Negative': {'Neuroticism': 0.9}},
    'disgust': {'Confusion': {'Conscientiousness': -0.6}},
    'anger': {'Anger': {'Agreeableness': -0.7, 'Directness': 0.8}},
    'neutrality': {'Neutral': {'Conscientiousness': 0.7, 'Measured': 0.6}}
}

# Preprocessing functions
def convert_to_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img

def preprocess_image(img: np.ndarray) -> np.ndarray:
    # Convert to float32 for normalization
    img = img.astype(np.float32) / 255.0
    img = cv2.resize(img, tuple(config['image_size']))
    img_array = np.expand_dims(img, axis=0)
    img_array = convert_to_rgb(img_array[0])
    return np.expand_dims(img_array, axis=0)

def preprocess_text(text: str) -> np.ndarray:
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=config['max_length'], padding='post', truncating='post')
    return padded

# Load models
try:
    image_model = load_model(config['model_paths']['image'])
    text_model = load_model(config['model_paths']['text'])
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# Inference functions
def predict_image_emotion(img: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
    img_data = preprocess_image(img)
    if img_data is None:
        return None, None
    img_pred = image_model.predict(img_data, verbose=0)
    img_emotion_idx = np.argmax(img_pred, axis=1)[0]
    img_confidence = float(np.max(img_pred)) * 100
    emotion = emotion_map.get(str(img_emotion_idx), 'unknown')
    print(f"Image emotion: {emotion}, confidence: {img_confidence}%")
    return emotion, img_confidence

def predict_text_emotion(text: str) -> Tuple[Optional[str], Optional[float]]:
    text_data = preprocess_text(text)
    if text_data is None:
        return None, None
    text_pred = text_model.predict(text_data, verbose=0)
    text_emotion_idx = np.argmax(text_pred, axis=1)[0]
    text_confidence = float(np.max(text_pred)) * 100
    emotion = emotion_map.get(str(text_emotion_idx), 'unknown')
    print(f"Text emotion: {emotion}, confidence: {text_confidence}%")
    return emotion, text_confidence

# Personality mapping
def map_to_personality(image_emotion: str, image_conf: float, text_emotion: str, text_conf: float) -> Dict[str, str]:
    if image_emotion == 'unknown' or text_emotion == 'unknown':
        return {'error': 'Unable to determine personality due to unknown emotions.'}

    combined_traits = {}
    total_conf = image_conf + text_conf

    for emotion, mappings in PERSONALITY_MAPPING.items():
        if image_emotion == emotion or text_emotion == emotion:
            for category, traits in mappings.items():
                weight = (image_conf if image_emotion == emotion else 0) + (text_conf if text_emotion == emotion else 0)
                for trait, score in traits.items():
                    combined_traits[trait] = combined_traits.get(trait, 0) + (score * weight / total_conf)



    # Find the dominant category by summing trait scores
    category_scores = {}
    for mappings in PERSONALITY_MAPPING.values():
        for category, traits in mappings.items():
            category_scores[category] = sum(combined_traits.get(t, 0) for t in traits.keys())
    dominant_category = max(category_scores.items(), key=lambda x: x[1])[0] if category_scores else 'Neutral'

    personality_traits = {k: v for k, v in combined_traits.items() if v > 0.5}

    perceived_qualities = {
        'Joy/Positive': 'an energetic and approachable person',
        'Sadness/Negative': 'a sensitive and introspective individual',
        'Anger': 'a direct and assertive person',
        'Confusion': 'a spontaneous and free-spirited individual',
        'Admiration': 'a creative and expressive person',
        'Neutral': 'a calm and reliable person'
    }
    perceived_quality = perceived_qualities.get(dominant_category, 'a balanced individual')

    return {
        'facial_observation': f"{image_emotion} (conf: {image_conf:.1f}%)",
        'speech_observation': f"{text_emotion} (conf: {text_conf:.1f}%)",
        'personality_traits': personality_traits,
        'perceived_quality': perceived_quality
    }

# Tkinter GUI
class PersonalityMirror:
    def __init__(self, root):
        self.root = root
        self.root.title("Personality Reflection Mirror")

        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            sys.exit(1)

        # GUI elements
        self.canvas = tk.Canvas(root, width=config['image_size'][0], height=config['image_size'][1])
        self.canvas.pack()

        self.text_entry = tk.Entry(root, width=50)
        self.text_entry.pack(pady=5)
        self.text_entry.insert(0, "Enter your text here...")

        self.submit_button = tk.Button(root, text="Analyze", command=self.analyze)
        self.submit_button.pack(pady=5)

        self.result_label = tk.Label(root, text="", wraplength=400)
        self.result_label.pack(pady=5)

        # Start video loop
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Process frame for prediction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_emotion, img_conf = predict_image_emotion(frame_rgb)
            if img_emotion:
                self.current_img_emotion = img_emotion
                self.current_img_conf = img_conf

            # Convert to PIL for display
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((config['image_size'][0], config['image_size'][1]), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(frame_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.root.after(10, self.update_frame)  # Update every 10ms

    def analyze(self):
        text_input = self.text_entry.get()
        if not text_input.strip():
            self.result_label.config(text="Please enter some text.")
            return

        text_emotion, text_conf = predict_text_emotion(text_input)
        if text_emotion and self.current_img_emotion:
            result = map_to_personality(self.current_img_emotion, self.current_img_conf, text_emotion, text_conf)
            output = (f"Your {result['facial_observation']} and {result['speech_observation']} suggest the following "
                      f"personality traits: {', '.join(f'{k}: {v:.2f}' for k, v in result['personality_traits'].items())}.")
            output += f"\nOthers might see you as {result['perceived_quality']}."
            self.result_label.config(text=output)
        else:
            self.result_label.config(text="Error: Could not determine emotions.")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = PersonalityMirror(root)
    root.mainloop()