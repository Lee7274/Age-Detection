#pip install pillow
#import os
#os.chdir(r'C:\Users\user9\assignment')  # Replace with your actual folder path

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from sklearn.preprocessing import StandardScaler
import joblib

# Load the pre-trained model and scaler
model = joblib.load('knn_age_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_image(image):
    """Preprocess the image before making a prediction."""
    img = np.array(image)  # Convert PIL image to numpy array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (64, 64))  # Resize to match training input size
    img = img.flatten()  # Flatten the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = scaler.transform(img)  # Normalize the image
    return img

def main():
    st.title("Age Detection System")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process and predict
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)
        st.write(f"Predicted Age: {prediction[0]}")

if __name__ == "__main__":
    main()
