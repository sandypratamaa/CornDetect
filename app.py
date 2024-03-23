import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename
import base64

# Initialize Flask app
app = Flask(__name__)
model = tf.keras.models.load_model("modelcorn.h5")
corndiseases_classes = ["Corn Common Rust", "Corn Gray Leaf Spot", "Corn Healthy", "Corn Northern Leaf Blight"]
IMG_SIZE = (299, 299)

# Function to predict image
def predict_image(image):
    # Resize image
    resized_image = image.resize(IMG_SIZE)
    # Convert image to numpy array
    img_array = np.expand_dims(resized_image, 0)
    # Predict class
    predictions = model.predict(img_array)
    pred_class = corndiseases_classes[np.argmax(predictions[0])]
    return pred_class

# Streamlit app
def streamlit_app():
    st.title("Corn Disease Detection")

    uploaded_file = st.file_uploader("Upload an image of corn leaf", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Check if uploaded file is an image
        if uploaded_file.type.startswith('image'):
            pred_class = predict_image(image)
            st.success(f"Predicted Class: {pred_class}")
        else:
            st.error("Please upload an image file.")

# Flask routes
@app.route('/')
def beranda():
    return render_template('index.html')

@app.route('/streamlit')
def run_streamlit():
    streamlit_app()
    return Response("Streamlit app running...")

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app)
