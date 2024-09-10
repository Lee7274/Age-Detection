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
    
    # Check if the image is already grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert to grayscale if the image has 3 channels
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif len(img.shape) == 2:
        # If the image is already grayscale, ensure it's in the correct shape
        pass
    else:
        raise ValueError("Unexpected image format")
    
    img = cv2.resize(img, (48, 48, 1))  # Resize to match training input size
    img = img.flatten()  # Flatten the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    print(f"Processed image shape: {img.shape}")
    
    # Check shape of img before scaling
    if img.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Image shape {img.shape} does not match scaler's expected shape.")
    
    img = scaler.transform(img)  # Normalize the image
    return img

def main():
    st.title("Age Detection System")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image and convert it to RGB
        image = Image.open(uploaded_file).convert('RGB')

        # Resize image for display (ensure display size matches expectations)
        display_size = (48, 48)  # Adjust size as needed
        image = image.resize(display_size, Image.ANTIALIAS)

        # Display the image
        st.image(image, caption='Uploaded Image', width=display_size[0], use_column_width=False)

        # Process and predict
        try:
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            st.write(f"Predicted Age: {prediction[0]}")
        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
