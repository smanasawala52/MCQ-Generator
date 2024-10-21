import base64
import io
import json
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from paddleocr import PaddleOCR


# Initialize PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # You can change the language if needed


def extract_features(image):
    # Convert the PIL Image to a NumPy array
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Extract features (e.g., SIFT, ORB)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_np, None)

    # Convert descriptors to a list of strings
    descriptor_strings = [base64.b64encode(descriptor).decode('utf-8') for descriptor in descriptors]

    return descriptor_strings


def analyze_image(image):
    # Convert the PIL Image to a NumPy array for PaddleOCR processing
    image_np = np.array(image)

    # Use PaddleOCR to extract text from the image
    result = ocr.ocr(image_np, cls=True)  # Perform OCR

    # Extracting the text from the OCR result
    extracted_text = ""
    for line in result:
        for box in line:
            extracted_text += f"{box[1][0]} "  # Append the recognized text

    return extracted_text.strip()


def main():
    # Streamlit UI
    st.title("Image Search Term Extractor (with PaddleOCR)")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Step 1: Analyze the image using PaddleOCR
        search_term = analyze_image(image)

        # Display the extracted search term
        st.write(f"Extracted Text: **{search_term}**")


if __name__ == "__main__":
    main()
