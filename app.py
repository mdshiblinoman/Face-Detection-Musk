from __future__ import annotations

import streamlit as st
from PIL import Image

from inference import load_model as load_model_uncached
from inference import predict


@st.cache_resource
def load_model():
    return load_model_uncached()


st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="centered")
st.title("😷 Face Mask Detection")
st.write("একটি ছবি upload করুন, model বলবে মুখে mask আছে কি না।")

try:
    model, class_names = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

uploaded_file = st.file_uploader("ছবি নির্বাচন করুন", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="আপনার ছবি", use_container_width=True)
    label, confidence = predict(model, image, class_names)
    if label == "with_mask":
        st.success(f"Mask detected — confidence: {confidence:.2%}")
    else:
        st.warning(f"No mask detected — confidence: {confidence:.2%}")
