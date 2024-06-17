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
from src.mcqgenerator.IteninaryGeneratorSingle import generate_iteninary_single_prompts, generate_poi_single_prompts,generate_iteninary_poi_single_prompts
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
quiz_generation_template_response_json = open_file_in_same_directory('..\\..\\response_jsons\\IteninaryResponse.json')

# Initialize Pinecone
pinecone = Pinecone(api_key=KT_PINECONE_API_KEY)
index_name = 'travelify'
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
# Function to create a grid layout
def create_grid(places, columns=4):
    rows = len(places) // columns + int(len(places) % columns > 0)
    selected_places = []
    for row in range(rows):
        cols = st.columns(columns)
        for col_idx, place_idx in enumerate(range(row * columns, min((row + 1) * columns, len(places)))):
            with cols[col_idx]:
                place = places[place_idx]
                # Display the image with a checkbox
                st.image(place["image_url"], caption=place["place"])
                if st.checkbox(place["place"], key=place["place"]):
                    selected_places.append(place)
                
                #st.image(places[place_idx]["image_url"], caption=places[place_idx]["place"])
    return selected_places

def seach_and_populate_poi(search_query):
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
        # Display the search results
        if search_results:
            places = []
            for city in search_results:
                for idx, package in enumerate(city["packages"]):
                    if "places_covered" in package:
                        for place in package["places_covered"]:
                            placeMap = {
                                        "place":place["place_covered_name"],
                                         "image_url":place["place_covered_image_url"][0]}
                            if not any(existing_place["place"] == placeMap["place"] and 
                                    existing_place["image_url"] == placeMap["image_url"] for existing_place in places):
                                #st.json(placeMap)
                                places.append(placeMap)
                st.header(city["city"])
                st.write(city["about"])
                st.subheader(f"Places to visit in {city['city']}")
                st.caption(f"Select places to create itinerary")
                selected_places = create_grid(places)
                if st.button("Create itinerary"):
                    if selected_places:
                        selected_places_names = ", ".join([place["place"] for place in selected_places])
                        st.write(f"Selected Places: {selected_places_names}")
                        with st.spinner('loading.....'):
                            try:
                                with get_openai_callback() as cb:
                                    generate_iteninary_prompts_temp = generate_iteninary_poi_single_prompts()
                                    result = generate_iteninary_prompts_temp({
                                        'text': quiz_data,
                                        'places': selected_places_names,
                                        'city': city,
                                        'response_json': json.dumps(quiz_generation_template_response_json),
                                    })
                            except Exception as e:
                                traceback.print_exception(type(e), e, e.__traceback__)
                                st.error("Error")
                            else:
                                print(result)
                                if isinstance(result, dict):
                                    quiz1 = result.get("quiz", None).replace("json", "").replace("```", "").replace("```", "")
                                    if quiz1 is not None:
                                        quiz = json.loads(quiz1)
                                        display_Iteninary_gpt(st, quiz)
                                else:
                                    st.write(result)
                    else:
                        st.write("No places selected.")
            return search_results
        else:
            st.write("No results found.")
            return None
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
        elif category_info == "point_of_interest":
            #First try getting data locally, of not found, search vai AI
            poi_data = seach_and_populate_poi(location)
            if poi_data is None:
                if quiz_data is not None:
                    with st.spinner('loading.....'):
                        poi_response_json = [{"place":"place_1","image_url": "image_url_1"}, {"place":"place_2","image_url": "image_url_2"}]
                        try:
                            with get_openai_callback() as cb:
                                generate_poi_prompts_temp = generate_poi_single_prompts()
                                result_poi = generate_poi_prompts_temp({
                                    'packages': quiz_data,
                                    'city': location,
                                    'poi_response_json': poi_response_json,
                                })
                        except Exception as e:
                            traceback.print_exception(type(e), e, e.__traceback__)
                            st.error("Error")
                            create_grid(result_poi_test)
                        else:
                            try:
                                #st.write(result_poi)
                                if isinstance(result_poi, dict):
                                    result_poi_temp = result_poi.get("point_of_intrests", None).replace("json", "").replace("```", "").replace("```", "")
                                    if result_poi_temp is not None:
                                        #st.write(result_poi_temp)
                                        result_poi = json.loads(result_poi_temp)
                                        #st.json(result_poi)
                                        create_grid(result_poi)
                                else:   
                                    st.write(result)
                            except Exception as e:
                                traceback.print_exception(type(e), e, e.__traceback__)
                                st.error("Error")
                                create_grid(result_poi_test)
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
            # First try local to fetch data
            poi_data = seach_and_populate_poi(location)
            if poi_data is None:
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
#create_grid(result_poi_test, columns=3)