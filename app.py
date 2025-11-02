import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
import os
import requests
# Load YOLOv8 segmentation model
from ultralytics.utils.plotting import colors

# Change default color palette globally
# colors([255, 128, 0])  # orange

st.set_page_config(
    page_title="i3L AI System",
    layout="wide",
    initial_sidebar_state="auto"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("i3LUniversity.png", 
             use_container_width=True)


st.markdown(
    "<h1 style='text-align: center;'>PVC Detection</h1>",
    unsafe_allow_html=True
)

model_path = "best_arrhythmia.pt"

if not os.path.exists(model_path):
    url = "https://huggingface.co/Sadrawi/ModelARR/resolve/main/best_arrhythmia.pt"
    with open(model_path, 'wb') as f:
        f.write(requests.get(url).content)

model = YOLO(model_path)



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

        img = img_np.copy()
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        names = model.names

        st.write(names)

        for (x1, y1, x2, y2) in boxes:
            color = (0, 255, 255)  # yellow
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
            cv2.putText(img, names[0], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        st.image(img, caption="Fixed color (manual draw)", use_container_width=True)

        # # Visualize the mask overlay
        # seg_img = results.plot(labels=False, 
        #                        conf=False)  # returns a numpy array with the segmentation mask overlaid

        # # Show result
        # st.image(seg_img, 
        #     caption="Detection Result", 
        #     use_container_width=True)


