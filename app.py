import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# Load model
model = tf.keras.models.load_model("lung_disease_model.h5")

# Sidebar Info
st.sidebar.title("About")
st.sidebar.info("👩‍⚕️ This app detects **Pneumonia** from chest X-ray using a pre-trained CNN model.")
st.sidebar.markdown("Made with ❤️ using Streamlit and TensorFlow")

# Set title
st.title("Lung Disease Detection from Chest X-ray")
st.write("Upload a chest X-ray image to detect Pneumonia.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB').resize((150, 150))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # Display result
    if prediction[0][0] > 0.5:
        st.error("Prediction: Pneumonia Detected 😷")
    else:
        st.success("Prediction: Normal Lungs 😊")
st.markdown("---")
st.markdown("<center><h4 style='color: gray;'>🔬 Powered by TensorFlow & Streamlit</h4></center>", unsafe_allow_html=True)

