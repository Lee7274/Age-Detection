pip install pillow
import os
os.chdir(r'C:\Users\user9\assignment')  # Replace with your actual folder path

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
import joblib

# Load the pre-trained model and scaler
model = joblib.load('knn_age_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess_image(image_path):
    """Preprocess the image before making a prediction."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (64, 64))  # Resize to match training input size
    img = img.flatten()  # Flatten the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = scaler.transform(img)  # Normalize the image
    return img

def upload_image():
    """Handle the image upload and prediction."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize the image for display
        img = ImageTk.PhotoImage(img)
        
        # Update the image preview
        preview_label.config(image=img)
        preview_label.image = img
        
        # Process and predict
        processed_img = preprocess_image(file_path)
        prediction = model.predict(processed_img)
        result_label.config(text=f"Predicted Age: {prediction[0]}")
    else:
        messagebox.showwarning("Warning", "No image selected!")

# Create the main window
root = tk.Tk()
root.title("Age Detection")

# Set the size of the window
root.geometry("400x500")

# Create and place a label for the title
title_label = tk.Label(root, text="Age Detection System", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

# Create and place an upload button
upload_btn = tk.Button(root, text="Upload Image", command=upload_image, font=("Helvetica", 12))
upload_btn.pack(pady=10)

# Create and place a label for the image preview
preview_label = tk.Label(root, text="Image Preview", font=("Helvetica", 12))
preview_label.pack(pady=10)

# Create and place a label for the prediction result
result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()
