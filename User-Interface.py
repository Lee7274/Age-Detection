#pip install pillow
#import os
#os.chdir(r'C:\Users\user9\assignment')  # Replace with your actual folder path
import streamlit as st
from PIL import Image

def main():
    st.title("Image Display System")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image
        image = Image.open(uploaded_file)

        # Resize image for display
        display_size = (48, 48)  # Adjust this if needed, based on the display size you want
        image = image.resize(display_size)

        # Display the image with a specific width
        st.image(image, caption='Uploaded Image', width=150)  # Adjust width as needed

if __name__ == "__main__":
    main()

