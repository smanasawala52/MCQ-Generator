import streamlit as st
from google.cloud import vision
from google.cloud.vision_v1 import types
from PIL import Image
import io
import pytesseract


# Function to classify the image using Google Cloud Vision
def classify_image(image_content):
    client = vision.ImageAnnotatorClient()

    image = types.Image(content=image_content)
    response = client.label_detection(image=image)
    st.write(response)
    labels = response.label_annotations
    st.write(labels)
    return labels


# Set up Streamlit app
st.title("Image Context Extraction with Google Vision API")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Load the image
    image = Image.open(uploaded_image)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    image_content = image_bytes.getvalue()

    # Extract text from the image using pytesseract
    extracted_text = pytesseract.image_to_string(image)

    # Display the extracted text
    st.subheader("Extracted Text:")
    st.write(extracted_text)

    # Classify the image using Google Vision API
    labels = classify_image(image_content)

    # Display the labels
    st.subheader("Image Labels:")
    for label in labels:
        st.write(f"{label.description} (Confidence: {label.score:.2f})")

    # Generate a description based on labels
    if labels:
        description = f"The image contains: {', '.join([label.description for label in labels])}."
    else:
        description = "No recognizable content found in the image."

    st.subheader("Image Description:")
    st.write(description)
