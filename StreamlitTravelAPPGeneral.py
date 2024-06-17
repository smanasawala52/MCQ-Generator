import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import json
import hashlib
import numpy as np
import requests
import openai
import os
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data_iteninary,display_Iteninary_gpt, open_file_in_same_directory

load_dotenv()
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
KT_PINECONE_API_KEY=os.getenv('KT_PINECONE_API_KEY')
KT_SERP_API_KEY=os.getenv('KT_SERP_API_KEY')
TRIPADVISOR_API_KEY=os.getenv('KT_TRIPADVISOR_API_KEY')
os.environ["PINECONE_API_KEY"]=KT_PINECONE_API_KEY

quiz_generation_template_response_json = open_file_in_same_directory('..\\..\\response_jsons\\IteninaryResponse.json')


index_name = 'travelify'
# Initialize Pinecone
pinecone = Pinecone(api_key=KT_PINECONE_API_KEY)
dimension = 8  # Ensure this matches the dimension used in data loading
index = pinecone.Index(index_name)

# OpenAI API Key
openai.api_key = OPENAI_API_KEY  # Replace with your OpenAI API key

# Helper Functions
def text_to_vector(text):
    hash_object = hashlib.sha256(text.encode('utf-8'))
    hash_digest = hash_object.hexdigest()
    return np.array([int(hash_digest[i:i+8], 16) for i in range(0, 64, 8)], dtype=np.float32)

def get_itinerary(number_of_days, city, subject):
    # Construct prompt
    prompt = f"You are an expert Itinerary maker. Given the above packages details, it is your job to \
    Most importantly, make use of time_of_day attribute to make sure trips are organized correctly. \
    create an itinerary for {number_of_days} days for customer looking for {city} holidays on {subject}. \
    Make sure days are not repeated and check all the days to be confirming the text as well. \
    Make sure if user wants to explore multiple cities, then first give itinerary for one city and then for other. \
    Include check-in and check-out details for each city, ensuring a smooth transition between cities. \
    Keep the last day for check-out only."

    quiz_generation_template = f"""
    Take a deep breath and work on this step by step.    
    You are an expert Iteninary maker. Given the above packages details, it is your job to \
    Most importantly, make use of time_of_day attribute to make sure trips are organized correctly.
    create a iteninary for {number_of_days} of days for customer looking for {city} holidays on {subject}. \
    Make sure days are not repeated and check all the days to be confirming the text as well. \
    Make sure if user wants to explore multiple cities, then first give iteninary for one city and then for other. \
    Include check-in and check-out details for each city, ensuring a smooth transition between cities. \
    Keep last day for check-out only. \
    Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
    Ensure to make {number_of_days} days.
    {quiz_generation_template_response_json}
    """

    #st.write(quiz_generation_template)


    vector = text_to_vector(quiz_generation_template).tolist()
    result = index.query(vector=vector, top_k=number_of_days)
    st.write(result)


    # Format the result
    itinerary_response = {}
    for i, match in enumerate(result['matches']):
        city_data = match['metadata']
        day = i + 1
        itinerary_response[str(day)] = {
            "day": day,
            "cities": [city_data.get('city', '')],
            "image_urls": city_data.get('image_url', []),
            "activity_titles": [city_data.get('activity', '')],
            "short_descriptions": [city_data.get('details', '')[:100]],  # First 100 chars as short description
            "long_descriptions": [city_data.get('details', '')]
        }
    return itinerary_response

def get_reviews(query):
    url = f"https://api.tripadvisor.com/api/reviews?query={query}&api_key=YOUR_TRIPADVISOR_API_KEY"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else {"error": "Failed to fetch reviews"}

def get_general_info(query):
    url = f"https://serpapi.com/search?q={query}&api_key={KT_SERP_API_KEY}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else {"error": "Failed to fetch information"}

def get_videos(query):
    url = f"https://serpapi.com/search?q={query}+youtube&api_key={KT_SERP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        video_urls = [video['link'] for video in data.get('video_results', [])]
        return video_urls
    return []

def classify_query(query):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that classifies queries into one of the following categories: itinerary, review, video, general_info."},
            {"role": "user", "content": f"Query: {query}\nCategory:"}
        ]
    )
    st.write(response)
    #category = response['choices'][0]['message']['content'].strip().lower()
    category = response.choices[0].message.content.strip()
    return category

def extract_itinerary_details(query):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that extracts details like number of days, holiday subject, and city from a query in json."},
            {"role": "user", "content": f"Extract the following details from the query: number of days, holiday subject, city in json.\nQuery: {query}\nDetails:"}
        ]
    )
    #details = response['choices'][0]['message']['content'].strip().split('\n')
    details = response.choices[0].message.content.strip()
    details=json.loads(details)
    st.write(details)
    return details

# Streamlit App
st.title("Travel Query Assistant")

search_query = st.text_input("Enter your query:")

if search_query:
    category = classify_query(search_query)
    
    if category == "itinerary":
        details = extract_itinerary_details(search_query)
        number_of_days = int(details.get('number_of_days', 1))
        subject = details.get('holiday_subject', '')
        city = details.get('city', '')
        itinerary = get_itinerary(number_of_days, city, subject)
        st.json(itinerary)
    elif category == "review":
        reviews = get_reviews(search_query)
        st.json(reviews)
    elif category == "video":
        videos = get_videos(search_query)
        for video in videos:
            st.video(video)
    else:
        general_info = get_general_info(search_query)
        st.json(general_info)
