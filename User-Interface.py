import numpy as np
import cv2
import joblib
from PIL import Image
import streamlit as st

# Load the pre-trained models and scaler
age_model = joblib.load('knn_age_model.pkl')
ethnicity_model = joblib.load('knn_ethnicity_model.pkl')
gender_model = joblib.load('knn_gender_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define mappings (update these as per your actual mappings)
gender_mapping = {0: "Male", 1: "Female"}
ethnicity_mapping = {0: "White", 1: "Black", 2: "Asian", 4: "Indian"}

def preprocess_image(image):
    """Preprocess the image before making a prediction."""
    img = np.array(image)  # Convert PIL image to NumPy array
    if len(img.shape) == 3:  # Convert to grayscale if it's a 3-channel image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize to 200x200 to match the model's input size
    resized_img = cv2.resize(img, (96, 128))

    # Flatten the resized image for the model input
    flattened_img = resized_img.flatten().reshape(1, -1)  # Shape it into (1, 40000) or (1, 12288)

    # Scale the flattened image using the scaler
    scaled_img = scaler.transform(flattened_img)
    
    return scaled_img

def main():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the uploaded image in its original size
        image = Image.open(uploaded_file).convert('RGB')  # Ensure color channels
        original_size = image.size  # Keep the original size for display

        # Display the original image in Streamlit
        st.image(image, caption=f'Uploaded Image (original size: {original_size})', use_column_width=True)

        try:
            # Preprocess the image
            processed_img = preprocess_image(image)

            # Make predictions for age, ethnicity, and gender
            age_prediction = age_model.predict(processed_img)
            ethnicity_prediction = ethnicity_model.predict(processed_img)
            gender_prediction = gender_model.predict(processed_img)

            # Convert numeric predictions to strings
            gender_str = gender_mapping.get(gender_prediction[0], "Unknown")
            ethnicity_str = ethnicity_mapping.get(ethnicity_prediction[0], "Unknown")

            # Display the predictions
            st.write(f"Predicted Age: {age_prediction[0]}")
            st.write(f"Predicted Ethnicity: {ethnicity_str}")
            st.write(f"Predicted Gender: {gender_str}")
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
