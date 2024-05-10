import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Fungsi untuk mendeteksi objek menggunakan OpenCV
def detect_objects(image):
    # Load model deteksi objek (misalnya YOLOv3)
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Resize image dan mendapatkan dimensi
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    height, width, channels = image.shape

    # Set input ke model
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Inisialisasi variabel
    class_ids = []
    confidences = []
    boxes = []

    # Loop melalui deteksi
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Mendapatkan koordinat kotak pembatas objek
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
    return image

# Daftar kelas objek (misalnya, COCO dataset)
classes = []
with open("./coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

st.title("Deteksi Objek dengan Streamlit")

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca gambar
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Deteksi objek
    detected_image = detect_objects(img_array)

    # Tampilkan hasil
    st.image(detected_image, caption='Deteksi Objek', use_column_width=True)
