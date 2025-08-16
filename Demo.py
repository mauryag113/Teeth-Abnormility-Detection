import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import os

# Load model
model = YOLO("YOLOv8_Training/teeth_stain_detection7/weights/best.pt")

st.title("🦷 Teeth Location Detection Dashboard")

# Upload image or enter URL
option = st.radio("Choose Image Input Method:", ["Upload Image", "Enter Image URL"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

elif option == "Enter Image URL":
    image_url = st.text_input("Paste Image URL:")
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image, caption="Image from URL", use_column_width=True)
        except:
            st.error("Invalid image URL")

# Predict button
if st.button("🔍 Predict"):
    if image:
        results = model.predict(image, imgsz=640, conf=0.25)
        # Show annotated image
        for r in results:
            annotated_img = r.plot()
            st.image(annotated_img, caption="Detected Results", use_column_width=True)

        # Optional: Show prediction table
        boxes = results[0].boxes
        if boxes:
            st.subheader("📋 Prediction Details")
            for i, box in enumerate(boxes):
                st.write({
                    "Class": model.names[int(box.cls)],
                    "Confidence": float(box.conf),
                    "Box": box.xyxy.tolist()
                })
