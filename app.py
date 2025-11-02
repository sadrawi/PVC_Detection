import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
import os
import requests
# Load YOLOv8 segmentation model



image = Image.open('i3LUniversity.png')
model_path = "best_arrhythmia.pt"

if not os.path.exists(model_path):
    url = "https://huggingface.co/Sadrawi/ModelARR/resolve/main/best_arrhythmia.pt"
    with open(model_path, 'wb') as f:
        f.write(requests.get(url).content)

model = YOLO(model_path)

# Streamlit UI
st.title("i3L University PVC Detection")

uploaded_file = st.file_uploader("Upload an Image", 
    type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, 
        caption="Uploaded Image", 
        use_container_width=True)

    if st.button("Run Detection"):
        # Convert to numpy array
        img_np = np.array(image)

        # Run prediction
        results = model.predict(img_np, conf=0.5)[0]

        # Visualize the mask overlay
        seg_img = results.plot()  # returns a numpy array with the segmentation mask overlaid

        # Show result
        st.image(seg_img, 
            caption="Detection Result", 
            use_container_width=True)


