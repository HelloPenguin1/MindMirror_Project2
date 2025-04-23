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

# Load image-specific emotion mapping
try:
    with open('emotion_map.json', 'r') as f:
        image_emotion_map = json.load(f)
        # Convert string keys to integers for internal use
        image_emotion_map_int = {int(k): v for k, v in image_emotion_map.items()}
        print(f"Loaded image emotion map: {image_emotion_map}")
except FileNotFoundError:
    print("Error: emotion_map.json not found. Using default mapping.")
    image_emotion_map_int = {0: 'surprise', 1: 'fear', 2: 'disgust', 3: 'happiness', 4: 'sadness', 5: 'anger', 6: 'neutrality'}
    image_emotion_map = {str(k): v for k, v in image_emotion_map_int.items()}

# Load text-specific emotion mapping
try:
    with open('text_emotion_map.json', 'r') as f:
        text_emotion_map = json.load(f)
        # Convert string keys to integers for internal use
        text_emotion_map_int = {int(k): v for k, v in text_emotion_map.items()}
        print(f"Loaded text emotion map: {text_emotion_map}")
except FileNotFoundError:
    print("Error: text_emotion_map.json not found. Using default mapping.")
    text_emotion_map_int = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sadness', 5: 'surprise'}
    text_emotion_map = {str(k): v for k, v in text_emotion_map_int.items()}

# Load tokenizer
try:
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
except FileNotFoundError:
    print("Error: tokenizer.json not found. Please ensure it exists.")
    sys.exit(1)





SIMPLE_PERSONALITY_TRAITS = {
    'happiness': {
        'category': 'Positive/Outgoing',
        'traits': ['sociable', 'optimistic', 'friendly'],
        'description': 'a cheerful and outgoing person who enjoys connecting with others'
    },
    'surprise': {
        'category': 'Curious/Open',
        'traits': ['curious', 'adaptable', 'open-minded'],
        'description': 'someone who is curious and open to new experiences'
    },
    'fear': {
        'category': 'Cautious/Sensitive',
        'traits': ['careful', 'analytical', 'protective'],
        'description': 'a careful person who thinks before acting'
    },
    'disgust': {
        'category': 'Critical/Discerning',
        'traits': ['particular', 'principled', 'selective'],
        'description': 'someone with strong principles and clear boundaries'
    },
    'anger': {
        'category': 'Assertive/Direct',
        'traits': ['direct', 'confident', 'strong-willed'],
        'description': 'a straightforward person who stands up for their beliefs'
    },
    'sadness': {
        'category': 'Reflective/Deep',
        'traits': ['thoughtful', 'empathetic', 'deep-thinking'],
        'description': 'a sensitive and introspective individual'
    },
    'neutrality': {
        'category': 'Balanced/Stable',
        'traits': ['balanced', 'calm', 'steady'],
        'description': 'a stable and balanced individual'
    }
}

# Preprocessing functions
def convert_to_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img

def preprocess_image(img: np.ndarray) -> np.ndarray:
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
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# Inference functions with debug prints
def predict_image_emotion(img: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
    img_data = preprocess_image(img)
    if img_data is None:
        return None, None
    
    img_pred = image_model.predict(img_data, verbose=0)
    img_emotion_idx = np.argmax(img_pred, axis=1)[0]
    img_confidence = float(np.max(img_pred)) * 100
 
    
    emotion = image_emotion_map_int.get(img_emotion_idx, 'unknown')
    print(f"Image emotion detected: {emotion}, confidence: {img_confidence:.2f}%")
    return emotion, img_confidence






def predict_text_emotion(text: str) -> Tuple[Optional[str], Optional[float]]:
    text_data = preprocess_text(text)
    if text_data is None:
        return None, None
    
    text_pred = text_model.predict(text_data, verbose=0)
    text_emotion_idx = np.argmax(text_pred, axis=1)[0]
    text_confidence = float(np.max(text_pred)) * 100
    
    
    emotion = text_emotion_map_int.get(text_emotion_idx, 'unknown')
    print(f"Text emotion detected: {emotion}, confidence: {text_confidence:.2f}%")
    return emotion, text_confidence







# Simplified personality mapping function
def map_to_personality(image_emotion: str, image_conf: float, text_emotion: str, text_conf: float) -> Dict[str, str]:
    if image_emotion == 'unknown' or text_emotion == 'unknown':
        return {'error': 'Unable to determine personality.'}
    
    # Simply use the dominant emotion (higher confidence)
    dominant_emotion = image_emotion if image_conf > text_conf else text_emotion
    personality = SIMPLE_PERSONALITY_TRAITS.get(dominant_emotion, SIMPLE_PERSONALITY_TRAITS['neutrality'])
    
    secondary_emotion = text_emotion if image_conf > text_conf else image_emotion
    secondary_traits = SIMPLE_PERSONALITY_TRAITS.get(secondary_emotion, {}).get('traits', [])
    
    # Combine some traits from both emotions
    combined_traits = personality['traits']
    if secondary_traits:
        # Add one trait from secondary emotion
        combined_traits.append(secondary_traits[0])
    
    return {
        'facial_observation': f"{image_emotion} ({image_conf:.1f}%)",
        'speech_observation': f"{text_emotion} ({text_conf:.1f}%)",
        'personality_traits': combined_traits,
        'dominant_category': personality['category'],
        'perceived_quality': personality['description']
    }

# Tkinter GUI
class PersonalityMirror:
    def __init__(self, root):
        self.root = root
        self.root.title("Personality Reflection Mirror")
        self.root.configure(bg='#f0f0f0')

        # Initialize variables
        self.current_img_emotion = None
        self.current_img_conf = None
        self.photo = None
        self.captured_frame = None

        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Camera frame
        camera_frame = ttk.Frame(main_frame, borderwidth=2, relief="groove")
        camera_frame.grid(row=0, column=0, padx=10, pady=10)
        
        self.canvas = tk.Canvas(camera_frame, width=config['image_size'][0], height=config['image_size'][1])
        self.canvas.pack()

        # Capture button
        self.capture_btn = ttk.Button(main_frame, text="Capture Image", command=self.capture_image)
        self.capture_btn.grid(row=1, column=0, pady=5)

        # Text input section
        input_frame = ttk.Frame(main_frame, padding="5")
        input_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        ttk.Label(input_frame, text="Enter your thoughts:").pack(anchor="w")
        
        self.text_entry = ttk.Entry(input_frame, width=50)
        self.text_entry.pack(fill="x", pady=5)
        self.text_entry.insert(0, "Tell me about your day...")
        
        self.submit_button = ttk.Button(input_frame, text="Analyze Personality", command=self.analyze)
        self.submit_button.pack(pady=5)

        # Results section
        result_frame = ttk.Frame(main_frame, padding="5")
        result_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        
        ttk.Label(result_frame, text="Analysis Results:").pack(anchor="w")
        
        self.result_label = ttk.Label(result_frame, text="Capture an image and enter text to begin", 
                                     wraplength=400, justify="left")
        self.result_label.pack(fill="x", pady=5)

    def capture_image(self):
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.result_label.config(text="Error: Could not open camera")
            return

        # Capture single frame
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Process the captured frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.captured_frame = frame_rgb
            
            # Display the captured image
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((config['image_size'][0], config['image_size'][1]), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(frame_pil)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Process emotion from image
            self.current_img_emotion, self.current_img_conf = predict_image_emotion(frame_rgb)
            self.result_label.config(text="Image captured! Enter text and click Analyze")
        else:
            self.result_label.config(text="Error: Could not capture image")

    def analyze(self):
        if self.captured_frame is None:
            self.result_label.config(text="Please capture an image first")
            return

        text_input = self.text_entry.get()
        if not text_input.strip() or text_input == "Tell me about your day...":
            self.result_label.config(text="Please enter some text about yourself or your thoughts.")
            return

        text_emotion, text_conf = predict_text_emotion(text_input)
        
        if not self.current_img_emotion or not text_emotion:
            self.result_label.config(text="Error: Could not analyze emotions properly")
            return
            
        result = map_to_personality(self.current_img_emotion, self.current_img_conf, 
                                   text_emotion, text_conf)
        
        if 'error' in result:
            self.result_label.config(text=result['error'])
            return
            
        traits_str = ", ".join(result['personality_traits'])
        
        output = (
            f"Facial Expression: {result['facial_observation']}\n"
            f"Text Expression: {result['speech_observation']}\n\n"
            f"Dominant Personality Category: {result['dominant_category']}\n"
            f"Key Traits: {traits_str}\n\n"
            f"Personality Reflection: {result['perceived_quality']}"
        )
        
        self.result_label.config(text=output)

if __name__ == "__main__":
    root = tk.Tk()
    app = PersonalityMirror(root)
    root.mainloop()