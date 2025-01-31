import streamlit as st
import face_recognition
from PIL import Image
import numpy as np

st.title("Simple Face Detection")

uploaded_image = st.file_uploader("Choose a face image...", type=["jpg", "png", "jpeg"])  # Added jpeg

if uploaded_image is not None:
    try:  # Handle potential image processing errors
        image = Image.open(uploaded_image).convert("RGB") # Ensure RGB for face_recognition
        img_array = np.array(image)

        face_locations = face_recognition.face_locations(img_array)

        if face_locations:
            for (top, right, bottom, left) in face_locations:
                st.image(img_array, use_column_width=True)  # Display the original image
                st.write(f"Face detected at: Top={top}, Right={right}, Bottom={bottom}, Left={left}")

                # Draw rectangle (optional - requires OpenCV)
                # import cv2
                # img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # face_recognition uses RGB, OpenCV uses BGR
                # cv2.rectangle(img_cv2, (left, top), (right, bottom), (0, 255, 0), 2)
                # st.image(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB), use_column_width=True)  # Display with rectangle

                # Face Recognition (Add this if needed)
                # face_encodings = face_recognition.face_encodings(img_array, face_locations)
                # if face_encodings:
                #     # Compare face_encodings with known face encodings
                #     # ... (your face recognition logic)
                #     st.write("Face encodings extracted (add recognition logic here).")


        else:
            st.write("No faces found in the image.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
