import base64
import io
import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
KT_PINECONE_API_KEY = os.getenv('KT_PINECONE_API_KEY')
KT_SERP_API_KEY = os.getenv('KT_SERP_API_KEY')
TRIPADVISOR_API_KEY = os.getenv('TRIPADVISOR_API_KEY')
os.environ["PINECONE_API_KEY"] = KT_PINECONE_API_KEY

openai.api_key = OPENAI_API_KEY


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
    # Convert the PIL Image to a base64 string
    # Resize the image
    image = image.resize((75, 75))

    # Convert the PIL Image to a base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())

    # Ensure the string is under the maximum length
    if len(img_str) > 1048576:
        raise ValueError("Image is too large to send to OpenAI API")

    base64_image = img_str.decode('utf-8')

    response_json = {"searchTerm": "search_term_1"}
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that reads the image and returns the search term."},
            {"role": "user",
             "content": f"Query: What is the search_term of this base64 encoded image: {base64_image}? return only search term keywords."}
        ]
    )
    st.write(response)

    # Extract the generated text from the response
    text_response = response.json()['choices'][0]['text'].strip()
    return text_response


# Function to delete all files
def delete_all_files():
    # List all files
    files = openai.files.list()

    # Iterate over the files and delete each one
    for file in files.data:
        file_id = file.id
        openai.files.delete(file_id)
        print(f"Deleted file: {file_id}")


# Function to upload the image and get the file reference
def upload_image_to_openai(image):
    # Convert the image to bytes
    image_bytes = image.getvalue()
    delete_all_files()
    # Use the OpenAI API to upload the image
    response = openai.files.create(
        file=image_bytes,
        purpose="vision"
    )

    # Get the file reference ID
    return response.id


# Function to get image details using the file reference
def get_image_details_from_file_ref(file_ref):
    # Use the OpenAI API to get details about the image using the file reference
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that reads the image and returns the search term."},
            {"role": "user",
             "content": f"Query: are you able to view the uploaded file with file ID {file_ref}. Please take a deep "
                        f"breath and describe the image or provide the text within the image, also generating search "
                        f"terms based on that information."}
        ]
    )
    # st.write(response)

    # Assuming the API returns a description that can be used to deduce the search term
    description = response.choices[0].message.content.strip()

    return description


def main():
    # Streamlit UI
    st.title("Image Search Term Extractor")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        # st.image(image, caption='Uploaded Image', use_column_width=True)

        # Step 1: Upload the image to OpenAI and get the file reference
        file_ref = upload_image_to_openai(uploaded_image)

        # Step 2: Use the file reference to get the image details
        search_term = get_image_details_from_file_ref(file_ref)

        # Display the search term
        st.write(f"Search Term: **{search_term}**")
    # st.title("Image Analysis")

    # uploaded_file = st.file_uploader("Choose an image...")

    # if uploaded_file is not None:
    # Read the image using PIL
    #   image = Image.open(uploaded_file)

    # Extract features
    #  features = extract_features(image)
    # st.write(features)
    # Analyze the image
    # result = analyze_image(image)

    # st.write("The image is of:")
    # st.text(result)


if __name__ == "__main__":
    main()
