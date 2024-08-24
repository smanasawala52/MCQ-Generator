import io
from google.cloud import vision
from google.cloud.vision_v1 import types
from pytesseract import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
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

# Load the BLIP model and processor from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# Function to generate an image caption using BLIP
def generate_image_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption_gen_img = processor.decode(out[0], skip_special_tokens=True)
    return caption_gen_img


# Function to generate a search term using GPT-3 based on the image caption
def generate_search_term(caption_gen_img_in):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that reads the image and returns the search term."},
            {"role": "user",
             "content": f"Based on the following description, generate a search term: {caption_gen_img_in}"
             }
        ]
    )
    search_term_gen_img = response.choices[0].message.content.strip()
    return search_term_gen_img


def create_grid_hugg(places, columns=6, show_checkbox=True):
    rows = len(places) // columns + int(len(places) % columns > 0)
    selected_places = []
    for row in range(rows):
        cols = st.columns(columns)
        for col_idx, place_idx in enumerate(range(row * columns, min((row + 1) * columns, len(places)))):
            with cols[col_idx]:
                place = places[place_idx]
                # Display the image with a checkbox
                st.image(place, caption=f"Image {place_idx + 1}", use_column_width=True)


places = []


def classify_image(image_content):
    client = vision.ImageAnnotatorClient()

    image = types.Image(content=image_content)
    response = client.label_detection(image=image)
    # st.write(response)
    labels = response.label_annotations
    # st.write(labels)
    return labels
def huggingface_load_image_extractor():
    st.set_page_config(
        page_title="Travelify",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
        }
    )
    hide_img_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''

    st.markdown(hide_img_fs, unsafe_allow_html=True)

    # Streamlit UI
    st.title("Image Search Term Extractor")

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        places.append(image)
        create_grid_hugg(places, columns=8, show_checkbox=False)
        # st.image(image, caption='Uploaded Image', use_column_width=True)

        # Step 1: Generate an image caption using BLIP
        caption = generate_image_caption(image)
        st.write(f"Image Caption: **{caption}**")

        # Step 2: Generate a search term using GPT-3 based on the image caption
        search_term = generate_search_term(caption)

        # Display the search term
        st.write(f"Search Term: **{search_term}**")

        # Extract text from the image using pytesseract
        extracted_text = pytesseract.image_to_string(image)

        # Display the extracted text
        st.subheader("Extracted Text:")
        st.write(extracted_text)

        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image.format)
        image_content = image_bytes.getvalue()
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


huggingface_load_image_extractor()
