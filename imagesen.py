import streamlit as st
from PIL import Image
from deepface import DeepFace

# Streamlit UI
st.title("Image Emotion Detection App")
st.write("Upload an image to analyze its emotion.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Emotion analysis on the uploaded image
    if st.button("Analyze Emotion"):
        # Perform emotion analysis using deepface
        try:
            result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']  # Get the dominant emotion
            st.write(f"### Detected Emotion: {emotion}")
        except Exception as e:
            st.write(f"Error analyzing the image: {e}")
