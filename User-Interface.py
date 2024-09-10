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
  img = np.array(image)  # Convert PIL image to NumPy array
  print(f"Original image shape: {img.shape}")

  # Resize to match training input size (assuming RGB)
  img = cv2.resize(img, (48, 48))

  # Flatten the image
  img = img.flatten()

  # Add batch dimension
  img = np.expand_dims(img, axis=0)

  print(f"Processed image shape: {img.shape}")

  # Check shape of img before scaling
  if img.shape[1] != scaler.n_features_in_:
    raise ValueError(f"Image shape {img.shape} does not match scaler's expected shape.")

  # Normalize the image
  img = scaler.transform(img)
  return img

def main():
  uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

  if uploaded_file is not None:
    try:
      image = Image.open(uploaded_file)
      st.write(f"Original image size: {image.size}")

      # Resize image for prediction (assuming same size as training data)
      image = image.resize((48, 48))

      # Display the image
      st.image(image, caption='Uploaded Image', use_column_width=True)

      # Process and predict
      processed_img = preprocess_image(image)
      prediction = model.predict(processed_img)  # Replace with your model loading and prediction logic
      st.write(f"Predicted Age: {prediction[0]}")
    except ValueError as e:
      st.error(f"Error: {e}")

if __name__ == "__main__":
  main()
