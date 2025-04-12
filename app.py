# app.py

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# --- Load model and tokenizer ---
model = load_model('saved_models/fake_news_model.h5')

# Load your tokenizer (make sure you saved it during training)
with open('saved_models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 500  # same as during training

# --- Prediction function ---
def predict_news(news_text):
    seq = tokenizer.texts_to_sequences([news_text])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(padded)[0][0]
    label = "Real News" if prediction > 0.5 else "Fake News"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# --- Streamlit UI ---
st.title("📰 Fake News Detector by David Rizz")

user_input = st.text_area("Paste your news article here:")

if st.button("Analyze"):
    if user_input.strip():
        label, conf = predict_news(user_input)
        st.success(f"**Prediction:** {label} ({conf*100:.2f}% confidence)")
    else:
        st.warning("Please enter some text to analyze.")
