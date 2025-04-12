# Fake News Detector 📰🔍

A deep learning model built with TensorFlow to detect fake news based on article content.

## Features
- Text classification using LSTM
- Trained on combined `Fake.csv` and `True.csv` datasets
- Accuracy ~99%
- Deployment using Streamlit

## 🔧 Project Structure

- `notebooks/`: EDA and model training
- `saved_models/`: Saved Keras model and tokenizer
- `app.py`: Streamlit app for inference

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
