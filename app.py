import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('saved_models/nb_model.pkl')
vectorizer = joblib.load('saved_models/vectorizer.pkl')

# Streamlit UI
st.title("📰 Fake News Detector")
user_input = st.text_area("Enter a news article or headline:")

if st.button("Predict"):
    if user_input:
        transformed_text = vectorizer.transform([user_input])

        # Get prediction and probabilities
        prediction = model.predict(transformed_text)
        probabilities = model.predict_proba(transformed_text)[0]

        # Format
        fake_proba = probabilities[0] * 100
        real_proba = probabilities[1] * 100
        label = "🟢 Real" if prediction[0] == 1 else "🔴 Fake"

        # Output
        st.subheader(f"Prediction: {label}")
        st.write(f"🔴 Fake: {fake_proba:.2f}%")
        st.write(f"🟢 Real: {real_proba:.2f}%")
    else:
        st.warning("Please enter some text.")
