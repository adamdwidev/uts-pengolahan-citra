
import streamlit as st
import cv2
import numpy as np
 st.title("Crop Images")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x = st.number_input("X coordinate", value=0)
        y = st.number_input("Y coordinate", value=0)
        width = st.number_input("Width", value=image.shape[1])
        height = st.number_input("Height", value=image.shape[0])
        cropped_image = image[y:y+height, x:x+width]
        st.image(cropped_image, caption="Cropped Image", use_column_width=True)