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

# Load a pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define mappings (update these as per your actual mappings)
gender_mapping = {0: "Male", 1: "Female"}
ethnicity_mapping = {0: "White", 1: "Black", 2: "Asian", 4: "Indian"}

def preprocess_image(image):
    """Preprocess the image before making a prediction."""
    img = np.array(image)  # Convert PIL image to NumPy array

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if no faces were detected
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # Extract the first face region (x, y, w, h) and crop the image
    x, y, w, h = faces[0]
    face_img = gray_img[y:y+h, x:x+w]

    # Resize to 96x128 to match the model's input size
    resized_img = cv2.resize(face_img, (96, 128))

    # Flatten the resized image for the model input
    flattened_img = resized_img.flatten().reshape(1, -1)

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
        except ValueError as ve:
            st.error(f"Error: {ve}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
