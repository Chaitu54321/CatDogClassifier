import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cdomod2.h5")

model = load_model()

# Define class labels
class_labels = ['cat', 'dog', 'others']

# Image preprocessing function
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Streamlit UI
st.title("Cat-Dog-Other Classifier 🐱🐶🤖")
st.write("Upload an image to classify.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image)
    
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index]

    # Initial label from model
    final_label = class_labels[predicted_index]

    # Custom confidence threshold logic
    if final_label in ['dog', 'cat'] and confidence < 0.6:
        final_label = 'others'

    st.markdown(f"### Prediction: **{final_label}**")
   
