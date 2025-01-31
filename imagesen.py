import streamlit as st
from PIL import Image
import numpy as np
from deepface import DeepFace
import pandas as pd

st.set_page_config(page_title="Emotion Detection", layout="wide")

st.title("🎭 Emotion Detection App")
st.write("""
Upload an image containing a face, and the app will predict the emotion.
Supported emotions: Angry 😠, Disgust 🤢, Fear 😨, Happy 😃, Sad 😢, Surprise 😲, Neutral 😐
""")

# Custom CSS (Slightly improved)
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .st-bb {
        background-color: #f0f2f6;
    }
    .st-at {
        background-color: #4CAF50;
    }
    .st-emotion-table {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .dataframe { /* Ensure table has some height for styling */
        height: 200px; /* Adjust as needed */
    }
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="file_uploader")

if uploaded_file is not None:
    try:
        col1, col2 = st.columns(2)

        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)

        with col2:
            with st.spinner('Analyzing emotions...'):
                img_array = np.array(image)
                try:  # Inner try for DeepFace errors
                    results = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=True)
                except ValueError as e:
                    st.error(f"🚨 DeepFace Error: {e}")  # More specific error message
                    results = [] # Handle no face detection gracefully
                except Exception as e:
                    st.error(f"🚨 DeepFace Error: {e}")
                    results = []

            if results:  # Check if results is not empty
                result = results[0]  # Access the first (and usually only) result
                emotion_data = result['emotion']
                dominant_emotion = result['dominant_emotion'].capitalize()

                st.subheader("Analysis Results")

                df = pd.DataFrame(list(emotion_data.items()), columns=['Emotion', 'Probability (%)'])
                df['Probability (%)'] = df['Probability (%)'].round(2)
                st.dataframe(df.style.background_gradient(cmap='YlGnBu', subset=['Probability (%)']), use_container_width=True)

                emotion_emoji = {
                    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨', 'Happy': '😃',
                    'Sad': '😢', 'Surprise': '😲', 'Neutral': '😐'
                }
                st.success(f"**Dominant Emotion:** {dominant_emotion} {emotion_emoji.get(dominant_emotion, '')}")
            else:
                st.error("🚨 Could not analyze the face in the image.")  # Handle analysis failure


    except Exception as e:  # Catch other Streamlit or PIL errors
        st.error(f"🚨 An error occurred: {str(e)}")

st.markdown("---")
st.markdown("""
*This app uses [DeepFace](https://github.com/serengil/deepface) for emotion detection powered by TensorFlow.
Models trained on FER2013 dataset.*
""")
