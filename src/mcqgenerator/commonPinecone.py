from pinecone import Pinecone
import json
import requests
import openai
import os
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data_iteninary, display_Iteninary_gpt, open_file_in_same_directory
from src.mcqgenerator.IteninaryGeneratorSinglePinecone import generate_iteninary_single_prompts, generate_poi_single_prompts,generate_iteninary_poi_single_prompts
from langchain.callbacks import get_openai_callback
import wikipediaapi
import wikipedia
from fuzzywuzzy import process

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
KT_PINECONE_API_KEY = os.getenv('KT_PINECONE_API_KEY')
KT_SERP_API_KEY = os.getenv('KT_SERP_API_KEY')
TRIPADVISOR_API_KEY = os.getenv('TRIPADVISOR_API_KEY')
os.environ["PINECONE_API_KEY"] = KT_PINECONE_API_KEY
quiz_generation_template_response_json = open_file_in_same_directory('IteninaryResponse.json')

# Initialize Pinecone
pinecone = Pinecone(api_key=KT_PINECONE_API_KEY)
index_name = 'travel-project'
dimension = 8  # Ensure this matches the dimension used in data loading
index = pinecone.Index(index_name)

# OpenAI API Key
openai.api_key = OPENAI_API_KEY
result_poi_test=[{"place": "Motiongate Dubai","image_url": "https://media-cdn.tripadvisor.com/media/attractions-splice-spp-720x480/06/f7/b8/10.jpg"},{"place": "Aquaventure Waterpark","image_url": "https://assets.kerzner.com/api/public/content/ca19a689d92f402a8de6cab014ff320c?v=16c7e7b4&t=w2880"},{"place": "Lost chambers","image_url": "https://media-cdn.tripadvisor.com/media/attractions-splice-spp-720x480/07/a1/87/8a.jpg"},{"place": "Legoland Dubai","image_url": "https://lh3.googleusercontent.com/p/AF1QipOUmYDntK5fVZ6q16HmYCMMCyhzQXqsgpOG5ph7=s680-w680-h510"},{"place": "Legoland Water Park","image_url": "https://dynamic-media-cdn.tripadvisor.com/media/photo-o/1a/86/6a/41/caption.jpg?w=1200&h=-1&s=1"},{"place": "Dubai Eye","image_url": "https://media.tacdn.com/media/attractions-splice-spp-674x446/0c/06/75/d7.jpg"},{"place": "Burj Al Arab","image_url": "https://cf.bstatic.com/xdata/images/hotel/max1024x768/457389536.jpg?k=1b5b93930a67d2372178da9df3c090ec47fb1f5622f5664dde18c4362f31b355&o=&hp=1"},{"place": "Palm Jumeirah","image_url": "https://www.nakheel.com/images/nakheelcorporatelibraries/developments/projects/palm-jumeirahd5c941eb-64a9-42fa-8218-0bf74d94a376.jpg?sfvrsn=de950100_1"},{"place": "Atlantis Hotel","image_url": "https://cf.bstatic.com/xdata/images/hotel/max1024x768/534061289.jpg?k=699f7488c0dbe08c1371f9dedfef5344c8ed29f9268d19f40cded5ca3665a95c&o=&hp=1"},{"place": "World Islands","image_url": "https://xrealty.ae/wp-content/uploads/2024/01/9TNwKEsp-the-world-islands-1200x850-1.jpg"},{"place": "Museum of the Future","image_url": "https://upload.wikimedia.org/wikipedia/en/8/8c/Museum_of_the_future%2C_Dubai.jpeg"},{"place": "Hajar Mountains","image_url": "https://upload.wikimedia.org/wikipedia/commons/0/0d/Nakhal_Fort_1.jpg"},{"place": "Dubai Safari Park","image_url": "https://media-cdn.tripadvisor.com/media/attractions-splice-spp-720x480/12/28/75/c2.jpg"},{"place": "Ski Dubai","image_url": "https://dynamic-media-cdn.tripadvisor.com/media/photo-o/07/43/33/37/ski-dubai.jpg"},{"place": "Burj Khalifa","image_url": "https://upload.wikimedia.org/wikipedia/en/thumb/9/93/Burj_Khalifa.jpg/200px-Burj_Khalifa.jpg"},{"place": "Dubai Mall","image_url": "https://dynamic-media-cdn.tripadvisor.com/media/photo-o/0d/8c/0c/1a/taken-5-years-ago-it.jpg?w=1200&h=1200&s=1"},{"place": "The Dubai Fountain","image_url": "https://media.tacdn.com/media/attractions-splice-spp-674x446/0b/ee/fb/5d.jpg"},{"place": "Palm Jumeirah Beach","image_url": "https://www.propertyfinder.ae/blog/wp-content/uploads/2023/10/1-13.jpg"},{"place": "Miracle Garden","image_url": "https://dynamic-media-cdn.tripadvisor.com/media/photo-o/23/7d/82/16/caption.jpg?w=300&h=300&s=1"},{"place": "Global Village","image_url": "https://dynamic-media-cdn.tripadvisor.com/media/photo-o/2b/9d/1c/89/caption.jpg?w=300&h=300&s=1"},{"place": "Dubai Mall Zabeel","image_url": "https://bsbgroup.com/application/files/6415/8936/8586/DSC_4767_900px.jpg"},{"place": "Glow Garden","image_url": "https://dynamic-media-cdn.tripadvisor.com/media/photo-o/19/e6/9a/c9/dubai-garden-glow-season.jpg?w=1200&h=1200&s=1"},{"place": "Dubai Marina","image_url": "https://res.klook.com/image/upload/c_fill,w_750,h_560/q_80/w_80,x_15,y_15,g_south_west,l_Klook_water_br_trans_yhcmh3/activities/sq20ughmmfbhkhyompao.jpg"},{"place": "Dubai Desert","image_url": "https://dynamic-media-cdn.tripadvisor.com/media/photo-o/2b/9d/26/bd/caption.jpg?w=500&h=400&s=1"}]
# Load quiz data
def load_data(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

quiz_data = load_data('data.json')

# Utility functions to interact with external APIs
def get_reviews(query):
    url = f"https://api.tripadvisor.com/api/reviews?query={query}&api_key={TRIPADVISOR_API_KEY}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else {"error": "Failed to fetch reviews"}

def get_general_info(query):
    url = f"https://serpapi.com/search?engine=google_light&q={query}&api_key={KT_SERP_API_KEY}"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else {"error": "Failed to fetch information"}

def get_videos(query):
    url = f"https://serpapi.com/search?engine=google_videos&q={query}&api_key={KT_SERP_API_KEY}"
    #st.write(url)
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        video_urls = [video['link'] for video in data.get('video_results', [])]
        return video_urls
    return []

def classify_query(query):
    response_json={"category":"category_1","place_location_event": "place_1"}
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that classifies queries into one of the following categories: itinerary, review, video, general_info, point_of_intrest."},
            {"role": "user", "content": f"Query: {query}\ncategory and place/location/event in json in lower case {response_json}"}
        ]
    )
    #st.write(response)
    #category = response['choices'][0]['message']['content'].strip().lower()
    category = response.choices[0].message.content.strip().replace('\'','\"')
    #st.write(category)
    category=json.loads(category)
    #st.json(category)
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
    #st.write(details)
    return details

# Wikipedia API function
def get_wikipedia_info(query):    
    #st.write(query)
    results = wikipedia.search(query, results=1, suggestion=False)
    try:
        title = results[0]    
        if query is not None:
            wpage = wikipedia.page(title,auto_suggest=False,preload=True)
            #st.write(wpage.title)
            #st.write(wpage.summary)
            #st.write(wpage.images)
            #st.write(wpage.url)
            return {
                "title": wpage.title,
                "summary": wpage.summary,            
                "images": wpage.images,
                "url": wpage.url
            }
    except:
        # Create a Wikipedia object with the User-Agent header
        wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (kasim.trophy@gmail.com)', 'en')
        # Access Wikipedia page (replace 'query' with your desired search term)
        page = wiki_wiki.page(query)
        if page.exists():
            return {
                "title": page.title,
                "summary": page.summary,
                "images": [],
                "url": page.fullurl
            }
        else:
            return None
def seach_and_populate_poi_common(search_query):
    search_results = []
    city_names = [city["city"] for city in quiz_data]
    if search_query:
        search_query = search_query.lower()  # Convert the search query to lowercase
        # Use fuzzy matching to find the closest matches
        matches = process.extract(search_query, city_names, limit=1)
        # Extract matched city names
        matched_names = [match[0] for match in matches if match[1] > 70]  # Adjust the threshold as needed
        # Find the cities corresponding to the matched names
        search_results = [city for city in quiz_data if city["city"] in matched_names]
    return search_results