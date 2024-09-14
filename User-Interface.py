import numpy as np
import cv2
import joblib
from sklearn.preprocessing import StandardScaler

# Define a function to preprocess the images
def preprocess_image(image, img_size=(128, 96)):
    img = np.array(image)  # Convert PIL image to NumPy array
    if len(img.shape) == 3:  # Convert to grayscale if it's a 3-channel image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Resize to the specified dimensions
    resized_img = cv2.resize(img, img_size)
    
    # Flatten the resized image for the model input
    flattened_img = resized_img.flatten().reshape(1, -1)  # Shape it into (1, features)
    
    return flattened_img

# Load and refit the StandardScaler with the new image size
def refit_scaler(images, img_size=(128, 96)):
    flattened_images = np.array([preprocess_image(img, img_size=img_size).flatten() for img in images])
    scaler = StandardScaler()
    scaler.fit(flattened_images)
    return scaler

# Example usage
# Assuming 'train_images' is a list or array of training images
# scaler = refit_scaler(train_images, img_size=(128, 96))

# Save the refitted scaler
# joblib.dump(scaler, 'scaler.pkl')

# For prediction
def preprocess_and_scale_image(image, scaler, img_size=(128, 96)):
    flattened_img = preprocess_image(image, img_size)
    scaled_img = scaler.transform(flattened_img)
    return scaled_img

# Use this function for preprocessing and scaling in your Streamlit app
