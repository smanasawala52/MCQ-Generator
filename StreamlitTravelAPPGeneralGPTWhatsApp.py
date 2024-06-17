from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from src.mcqgenerator.common import *
import json
app = Flask(__name__)

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
            #st.header(city["city"])
            #st.write(city["about"])
            #st.subheader(f"Places to visit in {city['city']}")
            #st.caption(f"Select places to create itinerary")
        return search_results
    else:
        return None

def create_whatsapp_message(data):
    day = data["day"]
    cities = ", ".join(data["cities"])
    activities = "".join(f"- {title}" for title in data["activity_titles"])

    message = f"**Day {day} - {cities}**{activities}"
    return message

@app.route("/whatsapp", methods=['POST'])
def whatsapp_reply():
    incoming_msg = request.values.get('Body', '').strip().lower()
    print(incoming_msg)
    response = MessagingResponse()
    msg = response.message()

    try:
        category_info = classify_query(incoming_msg)
        category = category_info['category']
        location = category_info['place_location_event']

        if category == "itinerary":
            details = extract_itinerary_details(incoming_msg)
            number_of_days = int(details.get('number_of_days', 1))
            subject = details.get('holiday_subject', '')
            city = details.get('city', '')
            with get_openai_callback() as cb:
                generate_iteninary_prompts_temp = generate_iteninary_single_prompts()
                result = generate_iteninary_prompts_temp({
                    'text': quiz_data,
                    'number': number_of_days,
                    'subject': city,
                    'tone': subject,
                    'response_json': json.dumps(quiz_generation_template_response_json),
                })
                quiz1 = result.get("quiz", None).replace("json", "").replace("```", "").replace("```", "")
                if quiz1 is not None:
                    quiz = json.loads(quiz1)
                    # Loop through each day and create messages
                    reply = ''
                    for day, info in quiz.items():
                        message = create_whatsapp_message(info)
                        print(f"Day {day} Message:\n{message}")
                        reply += f"Day {day} Message:{message}"
                    reply += ''
                    print(reply)
                    msg.body(reply)
                    print(msg)
                    return str(response)
        elif category == "point_of_interest":
            # Add logic for points of interest
            pass
        elif category == "review":
            reviews = get_reviews(location)
            reply = json.dumps(reviews, indent=2)
        elif category == "video":
            video_urls = get_videos(location)
            reply = "Videos:\n" + "\n".join(video_urls)
        else:
            wiki_info = get_wikipedia_info(location)
            if wiki_info:
                reply = f"{wiki_info['title']}\n\n{wiki_info['summary']}\n\nMore info: {wiki_info['url']}"
            else:
                general_info = get_general_info(location)
                reply = json.dumps(general_info, indent=2)
    except Exception as e:
        reply = str(e)

    msg.body(reply)
    return str(response)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
