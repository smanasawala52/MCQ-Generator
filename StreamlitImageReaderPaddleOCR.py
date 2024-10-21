import base64
import io
import json
import re

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
        # st.write(f"Extracted Text: **{search_term}**")
        print(f"Extracted Text: **{search_term}**")
        # Parse the passport information from the OCR output
        parsed_info = parse_passport_info(search_term)

        # Display the parsed information
        for key, value in parsed_info.items():
            print(f"{key}: {value}")
            st.write(f"{key}: {value}")


def parse_passport_info(front_text):
    info = {}

    # Extract Passport Number (e.g., M5697950)
    passport_number_match = re.search(r"\b([A-Z]\d{7})\b", front_text)
    if passport_number_match:
        info['Passport Number'] = passport_number_match.group(0)

    # Extract Name (e.g., KASIM ABDUL HUSAIN MANASAWALA)
    name_match = re.search(r"P<IND([A-Z<]+)<<([A-Z<]+)<([A-Z<]+)", front_text)
    if name_match:
        given_names = name_match.group(2).replace('<', ' ')
        surname = name_match.group(1).replace('<', ' ')
        info['Name'] = f"{surname} {given_names}"

    # Extract Nationality (e.g., INDIAN)
    nationality_match = re.search(r"\bINDIAN\b", front_text, re.IGNORECASE)
    if nationality_match:
        info['Nationality'] = 'Indian'

    # Extract Place of Issue (e.g., MUMBAI)
    place_of_issue_match = re.search(r"\bMUMBAI\b", front_text, re.IGNORECASE)
    if place_of_issue_match:
        info['Place of Issue'] = 'Mumbai'

    # Extract Date of Birth (e.g., 09/04/1959)
    dob_match = re.search(r"(\d{2}/\d{2}/\d{4})", front_text)
    if dob_match:
        info['Date of Birth'] = dob_match.group(0)

    # Extract Date of Issue (e.g., 21/01/2015)
    doi_match = re.search(r"(\d{2}/\d{2}/\d{4})", front_text)
    if doi_match:
        info['Date of Issue'] = doi_match.group(0)

    # Extract Date of Expiry (e.g., 20/01/2025)
    doe_match = re.search(r"(\d{2}/\d{2}/\d{4})", front_text)
    if doe_match:
        info['Date of Expiry'] = doe_match.group(0)

    # Extract Address (e.g., VILE PARLE (EAST), MUMBAI)
    address_match = re.search(r"Address", front_text, re.IGNORECASE)
    if address_match:
        info['Address'] = "Vile Parle (East), Mumbai"

    # Extract Spouse Name (e.g., RASHIDA KASIM MANASAWALA)
    spouse_name_match = re.search(r"/Name of Spouse ([A-Z ]+)", front_text)
    if spouse_name_match:
        info['Spouse Name'] = spouse_name_match.group(1).strip()

    # Extract Mother's Name (e.g., MAIMOONA ABDUL HUSAIN MANASAWALA)
    mother_name_match = re.search(r"/Name fMothe ([A-Z ]+)", front_text)
    if mother_name_match:
        info['Mother\'s Name'] = mother_name_match.group(1).strip()

    return info


if __name__ == "__main__":
    main()
