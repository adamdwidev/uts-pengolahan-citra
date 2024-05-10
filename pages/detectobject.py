# Python In-built packages
from pathlib import Path
import PIL
import os
from collections import Counter

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_imgs = []

class_names_file = 'weights/class_names.txt'  # Replace with the actual path
with open(class_names_file, 'r') as f:
    class_names = f.read().splitlines()

# Create a dictionary mapping class IDs to class names
CLASS_NAMES_DICT = {i: name for i, name in enumerate(class_names)}

# If image is selected
if source_radio == settings.IMAGE:
    uploaded_images = st.sidebar.file_uploader(
        "Upload multiple images...", type=("jpg", "jpeg", "png", 'bmp', 'webp'), accept_multiple_files=True)

    if uploaded_images:
        source_imgs.extend(uploaded_images)

# CLASS_NAMES_DICT=

if source_imgs:
    detect_button = st.sidebar.button("Detect Objects for All Images")

    if detect_button:
        class_counts_all_images = {}  # Dictionary to store class counts for all images

        for i, source_img in enumerate(source_imgs, start=1):
            col1, col2 = st.columns(2)

            with col1:
                try:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(uploaded_image, caption="Uploaded Image",
                             use_column_width=True)
                except Exception as ex:
                    st.error("Error occurred while opening the image.")
                    st.error(ex)

            with col2:
                if source_img is None:
                    # Create a default detected image with a message
                    default_detected_image = PIL.Image.new(
                        'RGB', (500, 500), (255, 255, 255))
                    draw = PIL.ImageDraw.Draw(default_detected_image)
                    draw.text((50, 50), "No image uploaded and detected.",
                              fill=(0, 0, 0))
                    st.image(default_detected_image, caption='Detected Image',
                             use_column_width=True)
                else:
                    res = model.predict(uploaded_image, conf=confidence)
                    boxes = res[0].boxes
                    # Extract the names of detected objects
                    names = res[0].names
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image',
                             use_column_width=True)

                    try:
                        with st.expander(f"Image {i} Detection Results"):
                            object_count = len(boxes)
                            st.write(
                                f"Number of objects detected: {object_count}")

                            # Count and display object names
                            object_name_counts = {}
                            for name in names:
                                object_name = CLASS_NAMES_DICT.get(name, name)
                                object_name_counts[object_name] = object_name_counts.get(
                                    object_name, 0) + 1

                            st.write("Object Counts:")
                            for name, count in object_name_counts.items():
                                # Use st.text() to display results
                                st.text(f"{name}: {count}")

                            # Store class counts for this image in the dictionary
                            class_counts_all_images[f"Image {i}"] = object_name_counts.copy(
                            )

                    except Exception as ex:
                        st.write("No objects detected!")