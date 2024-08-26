import streamlit as st
import json
import pandas as pd
import os
import hashlib
import time
from uuid import uuid4
from dotenv import load_dotenv
import pinecone
from tqdm.auto import tqdm
import openai
from langchain_openai import OpenAIEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec
import numpy as np

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
KT_PINECONE_API_KEY = os.getenv('KT_PINECONE_API_KEY')
openai.api_key = OPENAI_API_KEY
index_name = "travel-project"

# Initialize Pinecone client
print("Initializing Pinecone client...")
pc = pinecone.Pinecone(
    api_key=KT_PINECONE_API_KEY  # Replace with your actual Pinecone API key
)
print("Pinecone client initialized.")

spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
)

batch_limit = 150

st.set_page_config(
    page_title="Traveling",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)


def generate_id(value):
    """Generate a unique ID by hashing the input value."""
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


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


def get_about(about, minified=False):
    if minified:
        return ''
    else:
        return about


def validate_embedding(embedding):
    if not np.all(np.isfinite(embedding)):
        raise ValueError("Embedding contains non-finite values (NaN, infinity, etc.)")
    if not np.all(np.abs(embedding) <= 1):
        raise ValueError("Embedding values exceed the expected range [-1, 1]")


# Convert JSON to a flat DataFrame
def json_to_df(data, minified=False):
    rows = []
    for city_entry in data:
        city = city_entry.get('city', '')
        city_id = generate_id(city)  # Generate unique city_id
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
                place_covered_id = generate_id(f"{city}_{place_name}")  # Generate unique place_covered_id
                place_images = place.get('place_covered_image_url', [])
                if not isinstance(image_urls, list):
                    image_urls = [image_urls]
                if not isinstance(place_images, list):
                    place_images = [place_images]
                for image_url in image_urls:
                    for place_image in place_images:
                        row = {
                            'city': city,
                            'city_id': city_id,  # Include city_id in the row
                            'about': get_about(about, minified),
                            'time_of_day': time_of_day,
                            'popularity_score': popularity_score,
                            'adventure_score': adventure_score,
                            'activity_type': activity_type,
                            'activity': activity,
                            'details': get_about(details, minified),
                            'image_url': image_url,
                            'place_covered_name': place_name,
                            'place_covered_id': place_covered_id,  # Include place_covered_id in the row
                            'place_covered_image_url': place_image
                        }
                        rows.append(row)

    return rows


# Convert DataFrame back to JSON
def df_to_json(df):
    data = []
    grouped = df.groupby(['city', 'about', 'city_id'])
    for (city, about, city_id), group in grouped:
        packages = []
        package_grouped = group.groupby(
            ['time_of_day', 'popularity_score', 'adventure_score', 'activity_type', 'activity', 'details'])
        for package_key, package_df in package_grouped:
            image_urls = package_df['image_url'].unique().tolist()
            places_covered_grouped = package_df.groupby(['place_covered_name', 'place_covered_id'])
            places_covered = []
            for (place_name, place_covered_id), place_df in places_covered_grouped:
                place_images = place_df['place_covered_image_url'].unique().tolist()
                places_covered.append({
                    'place_covered_name': place_name,
                    'place_covered_id': place_covered_id,  # Include place_covered_id in the JSON
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
            'city_id': city_id,  # Include city_id in the JSON
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
                place_name = st.text_input(f"Place {i + 1} Name")
                place_images = st.text_area(f"Place {i + 1} Image URLs (comma-separated)").split(',')
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
                st.success("New entry added successfully!")
                pinecone_load_data_app()


def tiktoken_len(text):
    return len(text)


def pinecone_load_data_app():
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    pc.create_index(
        index_name,
        dimension=1536,  # Updated dimension to 1536
        metric='cosine',
        spec=spec
    )

    # Wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    index = pc.Index(index_name)
    st.write(index)

    # Load data from the saved JSON file
    data = load_data()

    # Initialize OpenAI embeddings and text splitter
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    st.write("OpenAI embeddings initialized.")
    st.write(embeddings)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=10,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Initialize lists for storing texts and metadata
    texts_load_data = []
    metadata_load_data = []

    # Index the data
    for i, record in enumerate(tqdm(data)):
        # Base metadata for the main record
        base_metadata = {
            'city': record.get('city', ''),
            'city_id': record.get('city_id', ''),
            'place_name': record.get('place_name', ''),
            'country': record.get('country', ''),
            'activity_type': record.get('activity_type', ''),
        }

        # Process the 'details' field
        if 'details' in record and record['details']:
            record_texts = text_splitter.split_text(record['details'])
            record_metadata = [{
                "chunk": j, "text": text, **base_metadata
            } for j, text in enumerate(record_texts)]
            texts_load_data.extend(record_texts)
            metadata_load_data.extend(record_metadata)

        # Process nested 'packages' and 'places_covered' entries
        if 'packages' in record:
            for package in record['packages']:
                for place_covered in package.get('places_covered', []):
                    # Metadata specific to each place_covered entry
                    place_covered_metadata = {
                        'place_covered_name': place_covered.get('place_covered_name', ''),
                        'time_of_day': package.get('time_of_day', ''),
                        'popularity_score': package.get('popularity_score', -1),
                        'adventure_score': package.get('adventure_score', -1),
                        'activity_type': package.get('activity_type', ''),
                        'activity': package.get('activity', ''),
                        'details': package.get('details', ''),
                        'image_urls': package.get('image_url', []),
                        'places_covered': package.get('places_covered', []),
                        'place_name': place_covered.get('place_covered_name', ''),
                        'place_covered_id': place_covered.get('place_covered_id', ''),
                        'place_images': place_covered.get('place_covered_image_url', []),
                        **base_metadata
                    }

                    # Flatten the metadata to ensure all fields are acceptable
                    flattened_metadata = flatten_metadata(place_covered_metadata)

                    # Generate vectors for the textual content of each place covered
                    texts_load_data.extend(text_splitter.split_text(place_covered_metadata['details']))
                    metadata_load_data.extend([flattened_metadata for _ in
                                               range(len(text_splitter.split_text(place_covered_metadata['details'])))])

    # Once all the texts and metadata are gathered, you can upsert them into Pinecone
    upsert_to_pinecone(texts_load_data, metadata_load_data)


def flatten_metadata(metadata):
    """
    Flatten the metadata to ensure all values are strings, numbers, booleans, or lists of strings.
    Convert lists of dictionaries or other unsupported types to strings.
    """
    flattened_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            flattened_metadata[key] = str(value)  # Convert nested dicts/lists to a string
        else:
            flattened_metadata[key] = value
    return flattened_metadata


def upsert_to_pinecone(texts, metadata):
    # Initialize Pinecone Index
    index = pc.Index(index_name)

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

    # Generate embeddings and upsert into Pinecone
    for i in tqdm(range(0, len(texts), batch_limit)):
        i_end = min(len(texts), i + batch_limit)
        batch_texts = texts[i:i_end]
        batch_metadata = metadata[i:i_end]
        ids_batch = [str(uuid4()) for _ in range(len(batch_texts))]
        embeds = embeddings.embed_documents(batch_texts)
        to_upsert = [
            (ids_batch[j], embeds[j], batch_metadata[j])
            for j in range(len(batch_texts))
        ]
        index.upsert(vectors=to_upsert)

    st.success("Data successfully upserted to Pinecone.")

    # Create an embedding for the query string
    # query_text = "activities in Dubai"
    query_text = "activities in Dubai"
    query_vector = embeddings.embed_query(query_text)

    # Query the Pinecone Index
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    # # Check if results are not None and have valid entries
    # if results and hasattr(results, 'matches'):
    #     # Create a dictionary to store unique results
    #     unique_results = {}

    #     for result in results['matches']:
    #         # Extract relevant metadata
    #         place_covered_name = result['metadata'].get('place_covered_name', '')
    #         popularity_score = result['metadata'].get('popularity_score', -1)
    #         activity_type = result['metadata'].get('activity_type', '')
    #         activity = result['metadata'].get('activity', '')
    #         details = result['metadata'].get('details', '')
    #         result_id = result['metadata'].get('place_covered_id', '')

    #         # Avoid duplicates by checking if the place_covered_name has already been seen
    #         if place_covered_name not in unique_results:
    #             unique_results[place_covered_name] = {
    #                 'id': result_id,
    #                 'place_covered_name': place_covered_name,
    #                 'popularity_score': popularity_score,
    #                 'activity_type': activity_type,
    #                 'activity': activity,
    #                 'details': details,
    #                 'score': result['score']
    #             }

    #     # Convert dictionary to a list and sort by popularity_score in descending order
    #     sorted_results = sorted(unique_results.values(), key=lambda x: x['popularity_score'], reverse=True)

    #     # Display the sorted and unique results
    #     for result in sorted_results:
    #         st.write(f"ID: {result['id']}, Place: {result['place_covered_name']}, "
    #                 f"Popularity Score: {result['popularity_score']}, Activity Type: {result['activity_type']}, "
    #                 f"Activity: {result['activity']}, Details: {result['details']}")
    # else:
    #     st.warning("No results returned from Pinecone.")

    # # Check if results are not None and have valid entries
    # if results and hasattr(results, 'matches'):
    #     for result in results['matches']:
    #         # Extract metadata fields
    #         metadata = result['metadata']
    #         place_name = metadata.get('place_covered_name', 'N/A')
    #         popularity_score = metadata.get('popularity_score', 'N/A')
    #         activity_type = metadata.get('activity_type', 'N/A')
    #         activity = metadata.get('activity', 'N/A')
    #         details = metadata.get('details', 'N/A')
    #         id = metadata.get('place_covered_id', 'N/A')

    #         # Display the extracted information
    #         st.write(f"ID: {id}")
    #         st.write(f"Place Name: {place_name}")
    #         st.write(f"Popularity Score: {popularity_score}")
    #         st.write(f"Activity Type: {activity_type}")
    #         st.write(f"Activity: {activity}")
    #         st.write(f"Details: {details}")
    #         st.write("-" * 50)  # Separator for readability
    # else:
    #     st.warning("No results returned from Pinecone.")

    # # Check if results are not None and have valid entries
    # if results and hasattr(results, 'matches'):
    #     for result in results['matches']:
    #         st.write(f"ID: {result.id}, Score: {result.score}")
    # else:
    #     st.warning("No results returned from Pinecone.")

    # -------------------------------------------------------------------------------------------------------
    if results and hasattr(results, 'matches'):
        unique_results = {}
        for result in results['matches']:
            metadata = result['metadata']
            result_id = metadata.get('place_covered_id', 'N/A')

            # Avoid duplicates
            if result_id not in unique_results:
                unique_results[result_id] = {
                    'place_name': metadata.get('place_covered_name', 'N/A'),
                    'popularity_score': metadata.get('popularity_score', 'N/A'),
                    'activity_type': metadata.get('activity_type', 'N/A'),
                    'activity': metadata.get('activity', 'N/A'),
                    'details': metadata.get('details', 'N/A')
                }

        for result_id, data in unique_results.items():
            st.write(f"ID: {result_id}")
            st.write(f"Place Name: {data['place_name']}")
            st.write(f"Popularity Score: {data['popularity_score']}")
            st.write(f"Activity Type: {data['activity_type']}")
            st.write(f"Activity: {data['activity']}")
            st.write(f"Details: {data['details']}")
            st.write("-" * 50)
    else:
        st.warning("No results returned from Pinecone.")


# Run the app
initialize_load_data_app()
