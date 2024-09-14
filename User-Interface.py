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

   if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    faces = detect_face(image)

    if len(faces) == 0:
        st.write("No face detected in the image.")
    else:
        for (x, y, w, h) in faces:
            face_img = image.crop((x, y, x+w, y+h))
            img_array = preprocess_image(face_img)
            st.image(face_img, caption="Detected Face", use_column_width=True)

            if age_model and gender_model and race_model:
                age_prediction = age_model.predict(img_array)
                gender_prediction = gender_model.predict(img_array)
                race_prediction = race_model.predict(img_array)

                age_groups = ['0-8', '9-18', '19-39', '40-59', '60+']
                gender_classes = ['Male', 'Female']
                race_classes = ['White', 'Black', 'Asian', 'Indian']

                predicted_age = age_groups[np.argmax(age_prediction)]
                predicted_gender = gender_classes[round(gender_prediction[0][0])]
                predicted_race = race_classes[np.argmax(race_prediction)]

                st.write(f"Predicted Age Group: **{predicted_age}**")
                st.write(f"Predicted Gender: **{predicted_gender}**")
                st.write(f"Predicted Race: **{predicted_race}**")
            else:
                st.error("One or more models could not be loaded. Please check the model files.")
