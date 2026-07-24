import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Helmet Detection", layout="centered")
st.title("🪖 Helmet Detection using YOLOv8")
st.write("Upload an image to detect whether people are wearing helmets.")

@st.cache_resource
def load_model():
    return YOLO("runs/detect/train-2/weights/best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Detection"):
        with st.spinner("Detecting..."):
            results = model(np.array(image))
            annotated = results[0].plot()  # returns numpy array with boxes drawn

        st.image(annotated, caption="Detection Result", use_container_width=True)

        st.subheader("Detected Objects")
        boxes = results[0].boxes
        if len(boxes) == 0:
            st.write("No objects detected.")
        else:
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                st.write(f"**{cls_name}** — confidence: {conf:.2%}")