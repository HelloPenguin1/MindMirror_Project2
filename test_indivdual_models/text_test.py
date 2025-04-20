import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import json
from tkinter import Tk, Entry, Button, Label

# Load configuration
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found. Please ensure it exists.")
        exit(1)

# Load text-specific emotion mapping
def load_text_emotion_map():
    try:
        with open('text_emotion_map.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: text_emotion_map.json not found. Using default mapping.")
        return {"0": "anger", "1": "fear", "2": "joy", "3": "love", "4": "sadness", "5": "surprise"}

# Load tokenizer
def load_tokenizer():
    try:
        with open('tokenizer.json', 'r', encoding='utf-8') as f:
            tokenizer_json = json.load(f)
            return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    except FileNotFoundError:
        print("Error: tokenizer.json not found. Please ensure it exists.")
        exit(1)

# Preprocess text input
def preprocess_text(text: str, tokenizer, max_length: int) -> np.ndarray:
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')

# Load the text model
def load_text_model(model_path: str):
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"Error loading text model: {e}")
        exit(1)

# Main prediction function
def predict_emotion():
    config = load_config()
    text_emotion_map = load_text_emotion_map()
    tokenizer = load_tokenizer()
    text_model = load_text_model(config['model_paths']['text'])

    # Set up GUI
    root = Tk()
    root.title("Text Emotion Predictor")
    root.geometry("400x200")

    # Result label
    result_label = Label(root, text="Enter text to predict emotion", wraplength=350)
    result_label.pack(pady=20)

    # Handle text analysis
    def analyze_text():
        text_input = entry.get().strip()
        if not text_input:
            result_label.config(text="Please enter some text.")
            return

        # Process and predict
        text_data = preprocess_text(text_input, tokenizer, config['max_length'])
        prediction = text_model.predict(text_data, verbose=0)
        emotion_idx = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction)) * 100
        emotion = text_emotion_map.get(str(emotion_idx), 'unknown')

        # Update result
        result_label.config(text=f"Predicted emotion: {emotion}\nConfidence: {confidence:.2f}%")

    # Text entry field
    entry = Entry(root, width=50)
    entry.pack(pady=10)
    entry.insert(0, "Type your text here...")

    # Analyze button
    analyze_button = Button(root, text="Analyze", command=analyze_text)
    analyze_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    predict_emotion()