import streamlit as st
from PIL import Image
import pytesseract

# Set up Streamlit app
st.title("Image Content Extraction and Description")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Load the image
    image = Image.open(uploaded_image)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Extract text from the image using pytesseract
    extracted_text = pytesseract.image_to_string(image)

    # Display the extracted text
    st.subheader("Extracted Text:")
    st.write(extracted_text)

    # Provide a basic description based on extracted text
    if extracted_text:
        description = f"The image contains the following text: '{extracted_text.strip()}'"
    else:
        description = "No text found in the image."

    st.subheader("Image Description:")
    st.write(description)
