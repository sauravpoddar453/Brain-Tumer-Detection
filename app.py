import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

# Page config
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for a stunning UI
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    h1 {
        color: #38bdf8;
        text-align: center;
        font-family: 'Inter', sans-serif;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
    }
    h3 {
        color: #cbd5e1;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8 0%, #3b82f6 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.4);
    }
    .prediction-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    .prediction-title {
        color: #38bdf8;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .prediction-result {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .result-glioma { color: #f43f5e; text-shadow: 0 0 10px rgba(244, 63, 94, 0.5); }
    .result-meningioma { color: #a855f7; text-shadow: 0 0 10px rgba(168, 85, 247, 0.5); }
    .result-notumor { color: #22c55e; text-shadow: 0 0 10px rgba(34, 197, 94, 0.5); }
    .result-pituitary { color: #eab308; text-shadow: 0 0 10px rgba(234, 179, 8, 0.5); }
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Brain Tumor Detector")
st.markdown("### Upload an MRI scan to detect and classify brain tumors instantly.")

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

@st.cache_resource
def load_model():
    try:
        # Recreate a valid Keras 3 model archive dynamically
        import zipfile
        keras_file = "compiled_model.keras"
        if not os.path.exists(keras_file):
            with zipfile.ZipFile(keras_file, 'w') as zf:
                zf.write('config.json')
                zf.write('metadata.json')
                zf.write('model.weights.h5')
        
        model = tf.keras.models.load_model(keras_file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.warning("Please ensure the model files (config.json, metadata.json, model.weights.h5) are present.")
else:
    uploaded_file = st.file_uploader("Choose an MRI scan image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
            
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            with st.spinner("AI is analyzing the scan..."):
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions = model.predict(img_array)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = CLASS_NAMES[predicted_class_idx]
                confidence = predictions[0][predicted_class_idx] * 100
                
                css_class = f"result-{predicted_class.replace(' ', '').lower()}"
                
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-title">Diagnosis Result</div>
                    <div class="prediction-result {css_class}">{predicted_class}</div>
                    <div style="font-size: 18px; color: #94a3b8;">Confidence Level: <span style="color: white; font-weight: bold;">{confidence:.1f}%</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Detailed Probabilities")
                for i, class_name in enumerate(CLASS_NAMES):
                    st.progress(float(predictions[0][i]), text=f"{class_name}: {predictions[0][i]*100:.1f}%")

