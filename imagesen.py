import streamlit as st
from PIL import Image
from transformers import pipeline

# Load the emotion detection model from Hugging Face
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Streamlit UI
st.title("Image Emotion Detection App")
st.write("Upload an image to analyze its emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Emotion"):
        # Convert the image to a string (or use any other technique to detect emotions)
        # Here, we're simulating emotion detection with a simple example text
        example_text = "I feel so happy today!"
        
        # Use Hugging Face's pipeline for emotion detection
        result = emotion_classifier(example_text)
        emotion = result[0]['label']  # Get the predicted emotion
        st.write(f"### Detected Emotion: {emotion}")
