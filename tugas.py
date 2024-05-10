import streamlit as st
import opencv-python as cv2
import numpy as np

# st.page_link("home.py", label="Home", icon="🏚️")
# st.page_link("main.py", label="Page 1")
# st.page_link("pages/profile.py", label="Page 2")
# # st.page_link("pages/page_2.py", label="Page 2", icon="2️⃣", disabled=True)
# st.page_link("http://www.google.com", label="Google", icon="🌍")

def rgb_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv_image

def calculate_histogram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    return histogram

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    alpha = 1 + contrast / 127
    beta = brightness
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def find_contours(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(gray_image, 127, 255, 0)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    st.title("Image Manipulation Web App")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.sidebar.subheader("Image Manipulations")
        selected_option = st.sidebar.selectbox("Select an option", ["RGB to HSV", "Histogram", "Brightness and Contrast", "Contour"])

        if selected_option == "RGB to HSV":
            hsv_image = rgb_to_hsv(image)
            st.image(hsv_image, caption="HSV Image", use_column_width=True)

        elif selected_option == "Histogram":
            histogram = calculate_histogram(image)
            st.bar_chart(histogram)

        elif selected_option == "Brightness and Contrast":
            brightness = st.slider("Brightness", -100, 100, 0)
            contrast = st.slider("Contrast", -100, 100, 0)
            adjusted_image = adjust_brightness_contrast(image, brightness, contrast)
            st.image(adjusted_image, caption="Adjusted Image", use_column_width=True)

        elif selected_option == "Contour":
            contours = find_contours(image)
            image_with_contours = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
            st.image(image_with_contours, caption="Image with Contours", use_column_width=True)
if __name__ == "__main__":
    main()
