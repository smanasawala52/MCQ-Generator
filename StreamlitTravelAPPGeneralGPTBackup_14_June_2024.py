import streamlit as st
from pinecone import Pinecone
import json
import requests
import openai
import os
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data_iteninary, display_Iteninary_gpt, open_file_in_same_directory
from src.mcqgenerator.IteninaryGeneratorSingle import generate_iteninary_single_prompts
from langchain.callbacks import get_openai_callback
import wikipediaapi
import wikipedia

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
KT_PINECONE_API_KEY = os.getenv('KT_PINECONE_API_KEY')
KT_SERP_API_KEY = os.getenv('KT_SERP_API_KEY')
TRIPADVISOR_API_KEY = os.getenv('TRIPADVISOR_API_KEY')
os.environ["PINECONE_API_KEY"] = KT_PINECONE_API_KEY
quiz_generation_template_response_json = open_file_in_same_directory('..\\..\\response_jsons\\IteninaryResponse.json')

# Initialize Pinecone
pinecone = Pinecone(api_key=KT_PINECONE_API_KEY)
index_name = 'travelify'
dimension = 8  # Ensure this matches the dimension used in data loading
index = pinecone.Index(index_name)

# OpenAI API Key
openai.api_key = OPENAI_API_KEY

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
    with st.spinner('loading.....'):
        url = f"https://api.tripadvisor.com/api/reviews?query={query}&api_key={TRIPADVISOR_API_KEY}"
        response = requests.get(url)
        return response.json() if response.status_code == 200 else {"error": "Failed to fetch reviews"}

def get_general_info(query):
    with st.spinner('loading.....'):
        url = f"https://serpapi.com/search?engine=google_light&q={query}&api_key={KT_SERP_API_KEY}"
        response = requests.get(url)
        return response.json() if response.status_code == 200 else {"error": "Failed to fetch information"}

def get_videos(query):
    with st.spinner('loading.....'):
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
    with st.spinner('loading.....'):
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

# Generate the home page
def generate_home_page():
    st.title="Travel Query Assistant"
    search_query = st.text_input("Travel Query Assistant:")
    if search_query:
        classify_category = classify_query(search_query)
        category_info = classify_category['category']
        location = classify_category['place_location_event']
        if  category_info == "itinerary":
            if quiz_data is not None:
                details = extract_itinerary_details(search_query)
                number_of_days = int(details.get('number_of_days', 1))
                subject = details.get('holiday_subject', '')
                city = details.get('city', '')
                with st.spinner('loading.....'):
                    try:
                        with get_openai_callback() as cb:
                            generate_iteninary_prompts_temp = generate_iteninary_single_prompts()
                            result = generate_iteninary_prompts_temp({
                                'text': quiz_data,
                                'number': number_of_days,
                                'subject': city,
                                'tone': subject,
                                'response_json': json.dumps(quiz_generation_template_response_json),
                            })
                    except Exception as e:
                        traceback.print_exception(type(e), e, e.__traceback__)
                        st.error("Error")
                    else:
                        if isinstance(result, dict):
                            quiz1 = result.get("quiz", None).replace("json", "").replace("```", "").replace("```", "")
                            if quiz1 is not None:
                                quiz = json.loads(quiz1)
                                display_Iteninary_gpt(st, quiz)
                        else:
                            st.write(result)
        elif category_info == "review":
            reviews = get_reviews(location)
            st.json(reviews)
        elif category_info == "video":
            video_urls = get_videos(location)
            # Extract video URLs from potential video links (assuming a specific format)
            #st.json(video_urls)
            # Display the extracted video URLs
            if video_urls:
                st.header(f"Videos related to {location}")
                n = 3
                groups = []
                for i in range(0,len(video_urls), n):
                    groups.append(video_urls[i:i+n])
                #st.write(groups)
                cols = st.columns(n)
                for group in groups:
                    for i, video_url in enumerate(group):
                        cols[i].video(video_url)

                #for url in video_urls:
                    #video_url = f"- [{url.split('/')[-2]}]({url})"
                    #st.write(url)  # Display video title and link
                    #st.video(url)
            else:
                st.info("No YouTube video links found in the provided data.")
        else:
            # First try Wikipedia
            wiki_info = get_wikipedia_info(location)
            if wiki_info:
                st.header(wiki_info['title'])
                if wiki_info['images']:
                    n = 3
                    max_image=n if len(wiki_info['images']) > n else len(wiki_info['images'])
                    groups = []
                    for i in range(0,max_image, n):
                        groups.append(wiki_info['images'][i:i+n])
                    #st.write(groups)
                    cols = st.columns(n)
                    for group in groups:
                        for i, video_url in enumerate(group):
                            cols[i].image(video_url)
                st.markdown(wiki_info['summary'])
                st.markdown(f"[Read more on Wikipedia]({wiki_info['url']})")
            else:
                # Fall back to SERP API if Wikipedia info not found
                general_info = get_general_info(location)
                #st.json(general_info)
                if general_info is not None:
                    if general_info['knowledge_graph'] is not None:
                        data = general_info['knowledge_graph']
                        st.header(data["title"])
                        try:
                            if data["image"] is not None:
                                st.image(data['header_images'][0]["image"], use_column_width=False)
                            if data["header_images"] is not None and data["header_images"][0] is not None and data["header_images"][0]["image"] is not None:
                                st.image(data['header_images'][0]["image"], use_column_width=False)
                        except Exception as e:
                            print(e)
                        st.markdown(data["description"])
                        left_column, right_column = st.columns(2)
                        with left_column:
                            st.subheader("Key Information")
                            if data["directions"] is not None:
                                try: st.caption(data["directions"])
                                except: pass
                            if data["website"] is not None:
                                try: st.caption(data["website"])
                                except: pass
                            if data["address"] is not None:
                                try: st.caption(data["address"])
                                except: pass
                            if data["address"] is not None:
                                try: st.caption(data["address"])
                                except: pass
                            if data["phone"] is not None:
                                try: st.caption(data["phone"])
                                except: pass
                            if data["age"] is not None:
                                try: st.caption(data["age"])
                                except: pass
                            if data["population"] is not None:
                                try: st.caption(data["population"])
                                except: pass
                            if data["emirate"] is not None:
                                try: st.caption(data["emirate"])
                                except: pass
                            if data["metro_population"] is not None:
                                try: st.caption(data["metro_population"])
                                except: pass
                        with right_column:
                            st.subheader("Points of Interest")
                            try:
                                for point in data["points_of_interest"]:
                                    try:
                                        st.image(point["image"])
                                        st.write(f"[{point['name']}]({point['link']})")
                                    except: pass
                            except: pass

generate_home_page()
#itenary={'1': {'day': 1, 'cities': ['Dubai','Abu Dhabi'], 'image_urls': ['https://s3-ap-southeast-1.amazonaws.com/akbartravelsholidays/package-itinerary17133325843560DesertSafari400x400.jpg', 'https://media-cdn.tripadvisor.com/media/attractions-splice-spp-674x446/0a/0e/99/c9.jpg'], 'activity_titles': ['Arrival in Dubai', 'Hot Air Balloon Ride', 'Desert Safari with Dune Bashing'], 'short_descriptions': ['Check-in and welcome to Dubai.', 'Experience a thrilling hot air balloon ride over the Arabian sands.', 'Adventure-filled desert safari with dune bashing, entertainment shows, and BBQ dinner.'], 'long_descriptions': ['Marhabaan and welcome to Dubai! On your arrival at Dubai International Airport, you will be received by our representative and transferred to your hotel. Settle into your luxurious room and spend the day at leisure exploring one of the most glamorous cities in the world.', 'Want to see Dubai differently? Choose to float 4,000ft above the Arabian sands on a hot air balloon tour. Balloon Adventures Dubai has created a world first – where passengers get to share the skies (and basket) with falcons, as they experience a one-of-a-kind aerial adventure.', 'Later in the afternoon, we depart for a desert adventure on a 5-hour safari from Dubai. Travel by a comfortable 4x4 to a welcoming camp in the middle of the desert. You will sample some traditional Arabic coffee and dates before taking off on a thrilling dune ride across the sands. End the day with a starlit barbecue dinner back at the camp. As you feast on delicious Arabic dishes, watch a sequined belly dancing show and hypnotic ‘tanoura’ performance, with dancers spinning in time to traditional Emirati music.']}, '2': {'day': 2, 'cities': ['Dubai'], 'image_urls': ['https://s3-ap-southeast-1.amazonaws.com/akbartravelsholidays/package-itinerary17131828643751YellowBoatTourDubai400X40.jpg', 'https://media-cdn.tripadvisor.com/media/attractions-splice-spp-720x480/0a/46/2d/a2.jpg'], 'activity_titles': ['Yellow Boat Tour', 'Legoland Dubai and Legoland Water Park'], 'short_descriptions': ['Experience Dubai sightseeing with The Yellow Boats.', 'Ultimate family adventure at Legoland Dubai and Legoland Water Park.'], 'long_descriptions': ['Experience Dubai sightseeing with a twist with The Yellow Boats. Don a life jacket and cruise along the coast in an inflatable boat, with the skyline behind you and Palm Jumeirah ahead. Sail past Atlantis, The Palm and marvel at the iconic Burj Al Arab Jumeirah before setting your eyes on the beautiful palaces that line the beach, with photo opportunities between every wave.', "Visit Legoland Dubai for the ultimate family adventure suited for kids from two to 12 years of age. Enjoy over 40 Lego-themed rides and shows before you head out to splash, play and even customize your own Lego raft at Legoland Water Park. Continue building up the fun with a stay at the newly-opened Legoland Hotel – a first in the region. If you're at Dubai Parks and Resorts with little ones, also stop by Neon Galaxy, the city's latest indoor playworld enticing youngsters into space-themed learning and fun."]}, '3': {'day': 3, 'cities': ['Dubai'], 'image_urls': ['https://s3-ap-southeast-1.amazonaws.com/akbartravelsholidays/package-itinerary17133319463560Airplanedeparture400x400.jpg'], 'activity_titles': ['Check-out from Dubai'], 'short_descriptions': ['Breakfast check-out from Hotel and Departure from Dubai.'], 'long_descriptions': ['After breakfast we check-out from our hotel & transfer to airport to catch flight back home. We return home with bags full of souvenirs and happy memories.']}}
#display_Iteninary_gpt(st,itenary)