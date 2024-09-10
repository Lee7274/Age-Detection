#pip install pillow
#import os
#os.chdir(r'C:\Users\user9\assignment')  # Replace with your actual folder path


import numpy as np
import cv2
import joblib
from PIL import Image
import streamlit as st

# Load the pre-trained model and scaler
model = joblib.load('knn_age_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_image(image):
    """Preprocess the image before making a prediction."""
    img = np.array(image)  # Convert PIL image to NumPy array
    if len(img.shape) == 3:  # Convert to grayscale if it's a 3-channel image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize image to 48x48 for model input
    resized_img = cv2.resize(img, (48, 48))
    
    # Flatten and reshape image for the model/scaler
    resized_img = resized_img.flatten()
    resized_img = np.expand_dims(resized_img, axis=0)  # Add batch dimension
    
    # Check shape to match scaler input
    if resized_img.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Image shape {resized_img.shape} does not match scaler's expected shape.")
    
    resized_img = scaler.transform(resized_img)  # Normalize using the scaler
    return resized_img

def main():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the uploaded image in its original size
        image = Image.open(uploaded_file).convert('L')
        original_size = image.size  # Keep the original size for display
        
        st.image(image, caption='Uploaded Image', use_column_width=True)

        try:
            # Preprocess the image (resize to 48x48 for model input but keep original for display)
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            st.write(f"Predicted Age: {prediction[0]}")
        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
