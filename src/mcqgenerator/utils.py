import os
import PyPDF2
import json
import traceback
import streamlit as st
from streamlit_carousel import carousel
from PIL import Image
from io import BytesIO
import requests

def open_file_in_same_directory(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    with open(file_path, 'r') as file:
        content = file.read()

    return content

def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text=""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception("Error reading the PDF file")
    elif file.name.endswith(".json"):
        try:
            return json.load(file)
        except Exception as e:
            raise Exception("Error reading the JSON file")
    elif file.name.endswith(".txt"):
        try:
            return file.read().decode("utf-8")
        except Exception as e:
            raise Exception("Error reading the text file")
    else:
        raise Exception("Unsupported file format")
    
def get_table_data(quiz_dict):
    try:
        #quiz_dict = json.load(quiz_str)
        quiz_table_data= []
        for key,value in quiz_dict.items():
            mcq = value["mcq"]
            options = " | ".join(
                [
                    f"{option}: {option_value}"
                    for option, option_value in value["options"].items()
                ]
            )
            correct = value["correct"]
            quiz_table_data.append({"Q":key,"MCQ": mcq, "Choices": options, "Correct": correct})
        return quiz_table_data
    except Exception as e:
        traceback.print_exception(type(e),e, e.__traceback__)
        return False
def get_table_data_iteninary(quiz_dict):
    try:
        #quiz_dict = json.load(quiz_str)
        quiz_table_data= []
        for key,value in quiz_dict.items():
            #day = value["day"]
            #correct = value["correct"]
            quiz_table_data.append(value)
        return quiz_table_data
    except Exception as e:
        traceback.print_exception(type(e),e, e.__traceback__)
        return False
import streamlit as st
import requests  # Or use urllib

# Fetch image from URL (if needed)
def get_image(url):
    response = requests.get(url)
    return response.content
import streamlit as st
from PIL import Image
def display_Iteninary_gpt(st,itinerary):
    st.title="Travel Itinerary"
    st.markdown("""
        <style>
            .stDivider {margin-top: 0px; margin-bottom: 0px;}
            .stTabs [data-baseweb="tab-list"] button {padding: 0px 0px;}
            .stTabs [data-baseweb="tab"] {margin-right: 0px;}
            .stHeader {margin-bottom: 0px;}
            .stSubheader {margin-bottom: 0px;}
            .stMarkdown {margin-bottom: 0px;}
            h3 {padding-bottom: 0px; padding-top: 0px;}
            p {padding-bottom: 0px; padding-top: 0px; margin-bottom: 0px}
        </style>
    """, unsafe_allow_html=True)

    # Tabbed container for Overview and Itinerary
    tab1, tab2 = st.tabs(["Overview", "Itinerary"])

    # Overview Tab
    with tab1:
        #st.header("Overview")
        #st.write("Welcome to the travel itinerary overview. Below is a brief summary of your trip.")
        for day in itinerary.values():
            cols = st.columns([1, 4])
            with cols[0]:
                st.subheader(f"Day {day['day']} | {', '.join(ordered_set(day['cities']))}")
            with cols[1]:
                unique_activities = ordered_set(day['activity_titles'])
                activity_list_html = '<span style="color:red">' + ', '.join(unique_activities) + '</span>'
                st.markdown(activity_list_html, unsafe_allow_html=True)

                #st.image(day['image_urls'][0], width=400)
                st.caption(f"{', '.join(ordered_set(day['short_descriptions']))}")

            st.divider()
    # Itinerary Tab
    with tab2:
        #st.header("Detailed Itinerary")
        for day in itinerary.values():
            title = f"Day {day['day']} | {', '.join(ordered_set(day['cities']))}"
            st.markdown(title)
            #st.image(day['image_urls'][0], width=200)
            #st.write(day['long_descriptions'][0])
            cols = st.columns([1, 2])
            with cols[0]:
                #st.image(day['image_urls'][0], width=150)
                images = [{'img': url, 'title': '', 'text': ''} for url in ordered_set(day['image_urls'])]
                print(images)
                #carousel(images)  
                #st.image(day['image_urls'][0], width=150)
                #for img_url in ordered_set(day['image_urls']):
                    #st.image(img_url, width=100)
                carousel(key=day['day'],items=images,indicators=False, width=1, controls=False, slide=True,interval=1000,wrap=True)

                # Customize the carousel settings as needed
                
            with cols[1]:
                st.markdown(f"{' '.join(ordered_set(day['long_descriptions']))}")
            st.divider()

# Helper function to load and resize image
def load_image(url, width=150):
    print(url)
    #response = requests.get(url)
    #img = Image.open(BytesIO(response.content))
    #img = img.resize((width, int(img.height * width / img.width)))  # Maintain aspect ratio
    return url
def ordered_set(iterable):
  """
  This function returns a list while preserving the order of elements
  from the iterable while removing duplicates.
  """
  seen = set()
  result = []
  for value in iterable:
    if value not in seen:
      result.append(value)
      seen.add(value)
  return result

