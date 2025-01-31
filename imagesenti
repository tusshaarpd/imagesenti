import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Load a pre-trained model for image sentiment analysis (ResNet as a feature extractor)
model = models.resnet18(pretrained=True)
model.eval()

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to predict sentiment based on image
def predict_sentiment(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        score = torch.mean(output).item()
    
    if score > 0:
        return "Positive - All Good!"
    else:
        return "Negative - All Bad!"

# Streamlit UI
st.title("Image Sentiment Analysis App")
st.write("Upload an image to analyze its sentiment.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Sentiment"):
        sentiment = predict_sentiment(image)
        st.write(f"### Sentiment: {sentiment}")
