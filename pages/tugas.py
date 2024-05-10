import streamlit as st
import cv2
import numpy as np

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
            selected_option = st.sidebar.selectbox("Select an option", ["RGB to HSV", "Histogram", "Brightness and Contrast", "Contour", "Grayscale", "Blur", "Edge Detection", "Thresholding", "Rotate", "Resize", "Flip", "Crop"], index=None)

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

            elif selected_option == "Grayscale":
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)

            elif selected_option == "Blur":
                blur_image = cv2.GaussianBlur(image, (5, 5), 0)
                st.image(blur_image, caption="Blurred Image", use_column_width=True)

            elif selected_option == "Edge Detection":
                edges = cv2.Canny(image, 100, 200)
                st.image(edges, caption="Edge Detected Image", use_column_width=True)

            elif selected_option == "Thresholding":
                _, threshold_image = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 127, 255, cv2.THRESH_BINARY)
                st.image(threshold_image, caption="Thresholded Image", use_column_width=True)

            elif selected_option == "Rotate":
                angle = st.slider("Angle", -180, 180, 0)
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                st.image(rotated_image, caption="Rotated Image", use_column_width=True)

            elif selected_option == "Resize":
                width = st.number_input("Width", value=image.shape[1])
                height = st.number_input("Height", value=image.shape[0])
                resized_image = cv2.resize(image, (int(width), int(height)))
                st.image(resized_image, caption="Resized Image", use_column_width=True)

            elif selected_option == "Flip":
                flip_direction = st.radio("Flip Direction", ["Vertical", "Horizontal"])
                if flip_direction == "Vertical":
                    flipped_image = cv2.flip(image, 0)
                else:
                    flipped_image = cv2.flip(image, 1)
                st.image(flipped_image, caption="Flipped Image", use_column_width=True)

            elif selected_option == "Crop":
                x = st.number_input("X coordinate", value=0)
                y = st.number_input("Y coordinate", value=0)
                width = st.number_input("Width", value=image.shape[1])
                height = st.number_input("Height", value=image.shape[0])
                cropped_image = image[y:y+height, x:x+width]
                st.image(cropped_image, caption="Cropped Image", use_column_width=True)

            elif selected_option == "Remove Background":
                background_removed_image = remove_background(image)
                st.image(background_removed_image, caption="Background Removed Image", use_column_width=True)


if __name__ == "__main__":
    main()
