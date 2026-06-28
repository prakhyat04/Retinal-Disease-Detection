import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = tf.keras.models.load_model("retinal_model.h5")

classes = ["cataract", "normal"]

st.title("🧠 Retinal Disease Detection AI")
st.write("Upload an eye image and get prediction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])

    st.success(f"Prediction: {classes[predicted_class]}")