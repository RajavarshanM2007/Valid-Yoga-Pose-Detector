import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import pandas as pd

# --- Configuration ---
MODEL_PATH = 'yoga_pose_detector_resnet50.h5'
IMAGE_SIZE = (224, 224)
# IMPORTANT: Replace these with the actual names found by your training script
# The order must match the order Keras assigned during training (usually alphabetical).
CLASS_NAMES = ['downdog', 'goddess', 'plank', 'tree', 'warrior2'] 
NUM_CLASSES = len(CLASS_NAMES)


@st.cache_resource
def load_yoga_model():
    """Loads the trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Please run the training script first.")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img):
    """Resizes and normalizes the image for the model."""
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    # Normalize the same way as training data (0-255 to 0-1)
    img_array /= 255.0
    return img_array

def predict_pose(model, processed_img):
    """Makes a prediction and returns the class name and confidence."""
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class_name, confidence,predictions

# --- Streamlit Application Layout ---
st.title("🧘 Yoga Pose Detector")
st.markdown("Upload an image to classify one of the trained yoga poses.")

model = load_yoga_model()

if model is not None:
    uploaded_file = st.file_uploader(
        "Choose a PNG or JPEG image...", 
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # Add a placeholder for prediction progress
        with st.spinner('Analyzing Pose...'):
            # Preprocess, predict, and display results
            processed_img = preprocess_image(img)
            pose, confidence, predictions = predict_pose(model, processed_img)

        st.success("Analysis Complete!")
        if(confidence<0.75):
            st.error("Not a yoga pose")
        else:
            st.metric(
                label="Predicted Pose", 
                value=f"**{pose.upper()}**", 
                delta=f"Confidence: {confidence*100:.2f}%"
            )
            
            # Display the full list of confidences (optional)
            if(confidence<0.85):
                st.error("Not a Yoga Pose")
            else:
                st.subheader("Confidence Scores")
                conf_df = pd.DataFrame({
                    'Pose': CLASS_NAMES,
                    'Confidence': predictions[0]
                }).sort_values(by='Confidence', ascending=False)
                st.dataframe(conf_df, hide_index=True)