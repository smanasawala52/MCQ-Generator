import streamlit as st
import json
import pandas as pd
import os
import re
import openai
from langchain_text_splitters import RecursiveJsonSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import retrieval_qa
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from src.mcqgenerator.utils import ordered_set
import hashlib
import numpy as np
import requests

load_dotenv()
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
KT_PINECONE_API_KEY=os.getenv('KT_PINECONE_API_KEY')
splitter = RecursiveJsonSplitter(max_chunk_size=500)
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
pc = Pinecone(api_key=KT_PINECONE_API_KEY)
os.environ["PINECONE_API_KEY"]=KT_PINECONE_API_KEY
embedding=OpenAIEmbeddings()
index_name = "travelify"
openai.api_key = OPENAI_API_KEY

st.set_page_config(
    page_title="Travelify",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
    }
)

# Load JSON data
def load_data():
    try:
        with open('data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Save JSON data
def save_data(data):
    with open('data.json', 'w') as f:
        json.dump(data, f, indent=4)

def get_about(about,minified=False):
    if minified:
        return ''
    else:
        return about

# Convert JSON to a flat DataFrame
def json_to_df(data, minified=False):
    rows = []
    for city_entry in data:
        city = city_entry.get('city', '')
        about = city_entry.get('about', '')
        for package in city_entry.get('packages', []):
            time_of_day = package.get('time_of_day', '')
            popularity_score = package.get('popularity_score', -1)
            adventure_score = package.get('adventure_score', -1)
            activity_type = package.get('activity_type', '')
            activity = package.get('activity', '')
            details = package.get('details', '')
            image_urls = package.get('image_url', [])
            places_covered = package.get('places_covered', [])
            for place in places_covered:
                place_name = place.get('place_covered_name', '')
                place_images = place.get('place_covered_image_url', [])
                if not isinstance(image_urls, list):
                    image_urls = [image_urls]
                if not isinstance(place_images, list):
                    place_images = [place_images]
                for image_url in image_urls:
                    for place_image in place_images:
                        row = {
                            'city': city,
                            'about': get_about(about,minified),
                            'time_of_day': time_of_day,
                            'popularity_score': popularity_score,
                            'adventure_score': adventure_score,
                            'activity_type': activity_type,
                            'activity': activity,
                            'details': get_about(details,minified),
                            'image_url': image_url,
                            'place_covered_name': place_name,
                            'place_covered_image_url': place_image
                        }
                        rows.append(row)
    
    #return pd.DataFrame(rows)
    return rows

# Convert DataFrame back to JSON
def df_to_json(df):
    data = []
    grouped = df.groupby(['city', 'about'])
    for (city, about), group in grouped:
        packages = []
        package_grouped = group.groupby(['time_of_day', 'popularity_score', 'adventure_score', 'activity_type', 'activity', 'details'])
        for package_key, package_df in package_grouped:
            image_urls = package_df['image_url'].unique().tolist()
            places_covered_grouped = package_df.groupby(['place_covered_name'])
            places_covered = []
            for place_name, place_df in places_covered_grouped:
                place_images = place_df['place_covered_image_url'].unique().tolist()
                places_covered.append({
                    'place_covered_name': ', '.join(ordered_set(place_name)),
                    'place_covered_image_url': place_images
                })
            packages.append({
                'time_of_day': package_key[0],
                'popularity_score': package_key[1],
                'adventure_score': package_key[2],
                'activity_type': package_key[3],
                'activity': package_key[4],
                'details': package_key[5],
                'image_url': image_urls,
                'places_covered': places_covered
            })
        data.append({
            'city': city,
            'about': about,
            'packages': packages
        })
    return data

def initialize_load_data_app():
    # Initialize Streamlit app
    st.title("City Packages Manager")

    # Load data
    data = load_data()

    # Convert JSON to DataFrame
    #df = json_to_df(data)
    rows = json_to_df(data)
    df = pd.DataFrame(rows)

    # Debugging: Check if 'place_covered_name' column exists and contains valid data
    if 'place_covered_name' not in df.columns:
        st.error("'place_covered_name' column is missing in the DataFrame.")
        return
    
    if df['place_covered_name'].isnull().all():
        st.error("'place_covered_name' column is present but contains all null values.")
        return
    
    # Get unique city names for dropdown
    city_names = df['city'].unique().tolist()

    # Search dropdown for cities
    search_term_city = st.selectbox("Search for city", ["All"] + city_names)

    search_term_activity = st.text_input("Search for activity")

    # Filter DataFrame based on search term
    filtered_df = df
    if search_term_city != "All":
        filtered_df = filtered_df[filtered_df['city'] == search_term_city]
    
    if search_term_activity:
        filtered_df = filtered_df[
            filtered_df['activity'].str.contains(search_term_activity, case=False)
        ]

    # Display filtered data in a table with editable fields
    edited_df = st.data_editor(filtered_df, num_rows="dynamic")

    # Save changes
    if st.button('Save Changes'):
        # Apply edits to the full DataFrame
        df.update(edited_df)
        # Convert DataFrame back to JSON
        updated_data = df_to_json(df)
        save_data(updated_data)
        st.success("Changes saved successfully!")
        pinecone_load_data_app()

    # Add new entry
    with st.expander("Add New Entry"):
        with st.form("add_form"):
            new_city = st.text_input("City")
            new_about = st.text_area("About")
            new_time_of_day = st.selectbox("Time of Day", ["MORNING TO NOON", "NOON TO EVENING", "FULL DAY"])
            new_popularity_score = st.number_input("Popularity Score", value=-1)
            new_adventure_score = st.number_input("Adventure Score", value=-1)
            new_activity_type = st.text_input("Activity Type")
            new_activity = st.text_input("Activity")
            new_details = st.text_area("Details")
            new_image_urls = st.text_area("Image URLs (comma-separated)").split(',')
            
            new_places_covered = {}
            num_places = st.number_input("Number of Places Covered", value=1, min_value=1)
            for i in range(num_places):
                place_name = st.text_input(f"Place {i+1} Name")
                place_images = st.text_area(f"Place {i+1} Image URLs (comma-separated)").split(',')
                if place_name.strip():
                    new_places_covered[place_name.strip()] = [img.strip() for img in place_images if img.strip()]

            add_submitted = st.form_submit_button("Add")
            if add_submitted:
                places_covered_list = []
                for place_name, place_images in new_places_covered.items():
                    places_covered_list.append({
                        'place_covered_name': place_name,
                        'place_covered_image_url': place_images
                    })
                new_entry = {
                    'city': new_city,
                    'about': new_about,
                    'time_of_day': new_time_of_day,
                    'popularity_score': new_popularity_score,
                    'adventure_score': new_adventure_score,
                    'activity_type': new_activity_type,
                    'activity': new_activity,
                    'details': new_details,
                    'image_url': [url.strip() for url in new_image_urls if url.strip()],
                    'places_covered': places_covered_list
                }
                for image_url in new_entry['image_url']:
                    for place in new_entry['places_covered']:
                        for place_image in place['place_covered_image_url']:
                            new_row = new_entry.copy()
                            new_row['image_url'] = image_url
                            new_row['place_covered_name'] = place['place_covered_name']
                            new_row['place_covered_image_url'] = place_image
                            df = df.append(new_row, ignore_index=True)
                updated_data = df_to_json(df)
                save_data(updated_data)
                st.success(f"New entry for {new_city} added successfully!")

    st.write("The data.json file has been updated. Please reload the app to see the changes.")

# Convert text to a vector (example using hash)
def text_to_vector(text):
    # For simplicity, we are using hash; you can replace this with a more appropriate embedding
    import hashlib
    import numpy as np
    hash_object = hashlib.sha256(text.encode('utf-8'))
    hash_digest = hash_object.hexdigest()
    return np.array([int(hash_digest[i:i+8], 16) for i in range(0, 64, 8)], dtype=np.float32)

# Prepare data for Pinecone
def prepare_data(data):
    vectors = []
    for city_entry in data:
        city = city_entry.get('city', '')
        about = city_entry.get('about', '')
        for package in city_entry.get('packages', []):
            time_of_day = package.get('time_of_day', '')
            popularity_score = package.get('popularity_score', -1)
            adventure_score = package.get('adventure_score', -1)
            activity_type = package.get('activity_type', '')
            activity = package.get('activity', '')
            details = package.get('details', '')
            image_urls = package.get('image_url', [])
            places_covered = package.get('places_covered', [])
            for place in places_covered:
                place_name = place.get('place_covered_name', '')
                place_images = place.get('place_covered_image_url', [])
                if not isinstance(image_urls, list):
                    image_urls = [image_urls]
                if not isinstance(place_images, list):
                    place_images = [place_images]
                for image_url in image_urls:
                    for place_image in place_images:
                        # Use the data to create a vector
                        vector_data = f"{city} {time_of_day} {activity_type} {activity} {details} {image_url} {place_name} {place_image}"
                        vector = text_to_vector(vector_data)
                        vectors.append((str(hash(vector_data)), vector.tolist()))  # (id, vector)
    return vectors

# Insert data into Pinecone
def insert_data(index, vectors):
    for vector_id, vector in vectors:
        index.upsert([(vector_id, vector)])

def pinecone_load_data_app():
    st.write("Sending JSON Data to Pinecone")
    # Add your Streamlit components here
    json_data = load_data()
    if json_data is not None:
        rows = json_to_df(json_data, True)
        #st.write(rows[1])
        docs = splitter.create_documents(texts=[rows],convert_lists=True)
        #st.write(docs[1])
        try:
            pc.delete_index(index_name)
        except Exception as e:
            print(e)

        pc.create_index(
            name=index_name,
            dimension=1536,  # Replace with your model dimensions
            metric="cosine",  # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        index = pc.Index(index_name)
        docsearch = PineconeVectorStore.from_documents(docs,  embedding=embedding, index_name=index_name)
        query="activity in Dubai"
        retriever = docsearch.as_retriever()
        docs_retriever = retriever.invoke(query)
        st.write(docs_retriever)
        #docs2 = docsearch.similarity_search(query)
        #st.write(docs2)
        #qa = retrieval_qa.from_chain_type(llm=openai, chain_type="stuff", retriever=docsearch.as_retriever())
        #result_ret_qa = qa.run(query)
        #st.write(result_ret_qa)

# Function to generate embeddings and upsert them into Pinecone
def sanitize_string(s):
    # Remove special characters and replace spaces with underscores
    s = re.sub(r'\W+', '', s)
    s = s.replace(' ', '_')
    return s.lower()

initialize_load_data_app()
