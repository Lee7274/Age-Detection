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

    # Resize to 48x48 for model input, keeping the original size unchanged for display
    resized_img = cv2.resize(img, (48, 48))

    # Flatten the resized image for the model input
    flattened_img = resized_img.flatten().reshape(1, -1)  # Shape it into (1, 2304)

    # Scale the flattened image using the scaler
    scaled_img = scaler.transform(flattened_img)
    
    return scaled_img

def main():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the uploaded image in its original size
        image = Image.open(uploaded_file).convert('L')
        original_size = image.size  # Keep the original size for display

        # Display the original image in Streamlit
        st.image(image, caption=f'Uploaded Image (original size: {original_size})', use_column_width=True)

        try:
            # Preprocess the image (resize to 48x48 for model input)
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            st.write(f"Predicted Age: {prediction[0]}")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
