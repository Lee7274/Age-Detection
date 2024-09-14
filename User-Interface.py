import numpy as np
import cv2
from PIL import Image
import streamlit as st

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    img = np.array(image)  # Convert PIL image to NumPy array
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

    # Adjust face detection parameters to improve sensitivity
    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.05,  # Smaller steps between scales for increased sensitivity
        minNeighbors=3,    # Decrease to reduce strictness
        minSize=(30, 30)   # Minimum size of detected face
    )
    return faces

def main():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')  # Ensure it's an RGB image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Check face detection
        faces = detect_faces(image)
        if len(faces) == 0:
            st.write("No face detected in the image.")
        else:
            st.write(f"{len(faces)} face(s) detected in the image.")
            st.write("Face coordinates:", faces)

if __name__ == "__main__":
    main()
