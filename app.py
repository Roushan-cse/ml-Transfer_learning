import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from utils import image_preprocessing,predictions

st.set_page_config(page_title="No Prediction", layout="centered")

# App title
st.title("🖼️ Upload Image for Prediction")

# Input field section
st.subheader("📤 Choose an Image File")
image_file = st.file_uploader("Upload your image (jpg/png/jpeg)", type=["jpg", "png", "jpeg"])

if image_file is not None:
    # Load and display image
    image = Image.open(image_file)
    st.image(image, caption="🧾 Uploaded Image", use_column_width=True)

    
    image_tensor = image_preprocessing(image)
    try:
        prediction = predictions(image_tensor)

    # Display prediction
        st.success(f"🔮 Result: {prediction}")
    except:
        st.write(f"This format of image is not accepted --help use gryscale")
else:
    st.info("📎 Waiting for image file…")