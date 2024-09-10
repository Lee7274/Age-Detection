#pip install pillow
#import os
#os.chdir(r'C:\Users\user9\assignment')  # Replace with your actual folder path


import numpy as np
import cv2
from PIL import Image
import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained scaler
scaler = joblib.load('scaler.joblib')

def preprocess_image(image):
    """Preprocess the image before making a prediction."""
    img = np.array(image)  # Convert PIL image to NumPy array
    print(f"Original image shape: {img.shape}")
    
    # Ensure the image is grayscale (1 channel)
    if len(img.shape) == 3 and img.shape[2] == 3:
        raise ValueError("Image is not grayscale")
    elif len(img.shape) == 2:
        pass  # Image is already grayscale
    else:
        raise ValueError("Unexpected image format")
    
    img = cv2.resize(img, (48, 48))  # Resize to match training input size
    img = img.flatten()  # Flatten the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    print(f"Processed image shape: {img.shape}")
    
    # Check shape of img before scaling
    if img.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Image shape {img.shape} does not match scaler's expected shape.")
    
    img = scaler.transform(img)  # Normalize the image
    return img

def main():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Open the image and convert it to grayscale
            image = Image.open(uploaded_file).convert('L')
            st.write(f"Original image size: {image.size}")

            # Resize image for display and prediction
            display_size = (48, 48)  # Adjust size as needed
            image = image.resize(display_size)
            st.write(f"Resized image size: {image.size}")

            # Display the image
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Process and predict
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            st.write(f"Predicted Age: {prediction[0]}")
        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
