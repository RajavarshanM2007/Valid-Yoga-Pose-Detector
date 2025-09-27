import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import pandas as pd

# --- Config ---
MODEL_PATH = 'yoga_pose_detector_resnet50.h5'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']

@st.cache_resource
def load_yoga_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Please run the training script first.")
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img):
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_pose(model, processed_img):
    pred = model.predict(processed_img)
    idx = np.argmax(pred, axis=1)[0]
    return CLASS_NAMES[idx], pred[0][idx], pred

# --- Streamlit UI ---
st.set_page_config(page_title="Yoga AI", page_icon="🧘", layout="centered")

st.markdown("""
<style>
    .header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        font-size: 3rem;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        color: transparent;
        text-align: center;
        margin-bottom: 0.1rem;
        user-select: none;
    }
    .description {
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.2rem;
        color: #444444;
        margin-bottom: 2rem;
        user-select: none;
    }
    .upload-area {
        background-color: #f5f5f5;
        border: 2px dashed #6a11cb;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        color: #555555;
        margin-bottom: 2rem;
        transition: border-color 0.3s ease;
    }
    .upload-area:hover {
        border-color: #2575fc;
        color: #000000;
    }
    .result-box {
        border: 2px solid #6a11cb;
        border-radius: 15px;
        padding: 20px;
        margin-top: 1.5rem;
        background-color: #fafafa;
        box-shadow: 3px 3px 8px rgba(101, 45, 236, 0.2);
    }
    .result-title {
        font-weight: 700;
        font-size: 1.75rem;
        color: #6a11cb;
        margin-bottom: 0.2rem;
        user-select: none;
    }
    .result-confidence {
        font-size: 1.1rem;
        color: #333333;
        user-select: none;
    }
    .warning {
        color: #cc0000;
        font-weight: 600;
        margin-top: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">Yoga AI</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Detect and classify yoga poses easily with AI</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(label='', type=['png', 'jpg', 'jpeg'], help='Upload an image file')

model = load_yoga_model()

if uploaded_file and model:
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner('Predicting pose...'):
        processed = preprocess_image(image_pil)
        pose, confidence, predictions = predict_pose(model, processed)

    # Display results
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="result-title">Predicted Pose: {pose.upper()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-confidence">Confidence: {confidence*100:.2f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if confidence < 0.75:
        st.markdown('<div class="warning">⚠️ Confidence is low. Try uploading a clearer image of a yoga pose.</div>', unsafe_allow_html=True)
    
    with st.expander('Confidence Scores for All Poses'):
        conf_df = pd.DataFrame({
            'Pose': CLASS_NAMES,
            'Confidence': predictions[0]
        }).sort_values(by='Confidence', ascending=False)
        st.dataframe(conf_df.style.background_gradient(cmap='Purples'), hide_index=True)

elif uploaded_file and not model:
    st.error('Model could not be loaded. Please check the model file.')

