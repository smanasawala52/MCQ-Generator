import io
import json

from google.cloud import vision
from google.cloud.vision_v1 import types
from pytesseract import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st
from PIL import Image
import os
import openai
from dotenv import load_dotenv
import streamlit as st
import speech_recognition as sr
from google.cloud import speech_v1p1beta1 as speech
import io
from fuzzywuzzy import fuzz

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
# Initialize recognizer
recognizer = sr.Recognizer()


# Function to record and recognize speech
def recognize_speech():
    with sr.Microphone() as source:
        st.write("Please say something...")

        if st.button("Stop Recording"):
            st.stop()
        audio = recognizer.listen(source)

        # Convert audio to binary data
        audio_data = io.BytesIO(audio.get_wav_data())

        # Load binary data into Google Speech-to-Text
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_data.read())
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )

        # Detects speech in the audio file
        response = client.recognize(config=config, audio=audio)

        # Extract and display the text
        for result in response.results:
            st.write(f"Transcript: {result.alternatives[0].transcript}")


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
    search_term_gen_img = response.choices[0].message.content.strip().replace("\"", "")
    return search_term_gen_img


def create_grid_hugg(places_grid, columns=6, show_checkbox=True):
    rows = len(places_grid) // columns + int(len(places_grid) % columns > 0)
    selected_places = []
    for row in range(rows):
        cols = st.columns(columns)
        for col_idx, place_idx in enumerate(range(row * columns, min((row + 1) * columns, len(places_grid)))):
            with cols[col_idx]:
                place = places_grid[place_idx]
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


def load_data(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def fuzzy_search(caption, search_term, image_description_google_vision, website_search_mapping_data, threshold=70):
    results = []

    for item in website_search_mapping_data:
        max_score = 0
        match_field = None

        # Check caption
        if item['caption']:
            score = fuzz.token_set_ratio(caption, item['caption'])
            if score > max_score:
                max_score = score
                match_field = 'caption'

        # Check search_term
        if item['search_term']:
            score = fuzz.token_set_ratio(search_term, item['search_term'])
            if score > max_score:
                max_score = score
                match_field = 'search_term'

        # Check image_description_google_vision
        if item['image_description_google_vision']:
            score = fuzz.token_set_ratio(image_description_google_vision, item['image_description_google_vision'])
            if score > max_score:
                max_score = score
                match_field = 'image_description_google_vision'

        # Add result if it meets the threshold
        if max_score >= threshold:
            results.append({
                'match_field': match_field,
                'match_score': max_score,
                'item': item
            })
    if results is not None and len(results) > 0:
        return results[0]['item']['url']
    else:
        return None


def process_image_search(image):
    # Step 1: Generate an image caption using BLIP
    caption = generate_image_caption(image)

    # Step 2: Generate a search term using GPT-3 based on the image caption
    search_term = generate_search_term(caption)

    # Step 3: Use Google Vision API to classify the image
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    image_content = image_bytes.getvalue()
    labels = classify_image(image_content)

    # Generate a description based on labels
    if labels:
        image_description_google_vision = f"The image contains: {', '.join([label.description for label in labels])}."
    else:
        image_description_google_vision = "No recognizable content found in the image."

    # Step 4: Fuzzy search to find the best matching URL
    website_search_mapping_data = load_data('website_search_mapping.json')
    url = fuzzy_search(caption, search_term, image_description_google_vision, website_search_mapping_data)

    # Default to a Vistaprint search if no exact match is found
    if url is None:
        if search_term:
            url = f"https://www.vistaprint.com/search?query={search_term}"
        elif caption:
            url = f"https://www.vistaprint.com/search?query={caption}"
        elif image_description_google_vision:
            url = f"https://www.vistaprint.com/search?query={image_description_google_vision}"

    return url


def huggingface_load_image_extractor_streamlit():
    st.set_page_config(
        page_title="Image Search Term Extractor",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    hide_img_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''
    st.markdown(hide_img_fs, unsafe_allow_html=True)

    uploaded_image = st.file_uploader("Upload an image to search on vistaprint.com site", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        search_result_url = process_image_search(image)

        # Display the result
        st.write(search_result_url)
        st.write(f'<iframe src="{search_result_url}" width="100%" height="7000"></iframe>', unsafe_allow_html=True)


def huggingface_load_image_extractor_flask(image):
    search_result_url = process_image_search(image)
    return search_result_url


huggingface_load_image_extractor_streamlit()
