import streamlit as st
from src.mcqgenerator.common import *
st.set_page_config(
    page_title="Travelify",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
    }
)
# Function to create a grid layout
def create_grid(places, columns=6):
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
    search_results = seach_and_populate_poi_common(search_query)
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
            with st.spinner('loading.....'):
                reviews = get_reviews(location)
                st.json(reviews)
        elif category_info == "video":
            with st.spinner('loading.....'):
                video_urls = get_videos(location)
                # Extract video URLs from potential video links (assuming a specific format)
                #st.json(video_urls)
                # Display the extracted video URLs
                if video_urls:
                    st.header(f"Videos related to {location}")
                    n = 5
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
            with st.spinner('loading.....'):
                # First try local to fetch data
                poi_data = seach_and_populate_poi(location)
            if poi_data is None:
                with st.spinner('loading.....'):
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
