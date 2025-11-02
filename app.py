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

        # ---- Adjustable parameters ----
        font_scale = 0.5      # 🔠 change this for font size
        font_thickness = 1     # line thickness of text
        box_thickness = 1      # line thickness of box
        color = (245, 73, 39)  # (B, G, R): yellow
        # -------------------------------

        for (x1, y1, x2, y2) in boxes:
            # Draw rectangle
            # cv2.rectangle(img, (x1, y1), (x2, y2), 
            #               color, box_thickness)

            # Put text (class label)
            text = "PVC"  # single class
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                        font_scale, 
                                        font_thickness)[0]
            text_x = x1 + 1
            text_y = max(y1 - 5, text_size[1] + 5)

            cv2.putText(
                img,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                font_thickness,
                cv2.LINE_AA
            )

        st.image(img, caption="Custom Font Size + Box", use_container_width=True)


