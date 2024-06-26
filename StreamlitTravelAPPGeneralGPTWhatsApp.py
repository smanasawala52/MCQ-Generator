from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from src.mcqgenerator.common import *
import json
import urllib.parse
app = Flask(__name__)

def search_and_populate_poi_text(search_query):
    """
    Searches for places of interest and returns details in text format.
    """
    search_results = seach_and_populate_poi_common(search_query)
    #print(search_results)
    if search_results:
        message = f"Found some interesting places for your search '{search_query}':\n"
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
                            places.append(placeMap)
                            message += f"Place: *{place['place_covered_name']}* \n"
        #print(message)
        return message[:1000]
    else:
        return "Sorry, couldn't find any places of interest for your search."

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
                    reply1 = ''
                    for day, info in quiz.items():
                        message = create_whatsapp_message(info)
                        print(f"Day {day} Message:\n{message}")
                        reply1 += f"Day {day} Message:{message}"
                    reply1 += ''
                    print(reply1)
                    msg.body(reply1[:1000])
                    print(msg)
                    return str(response)
        elif category == "point_of_interest":
            reply_point_of_intrest = search_and_populate_poi_text(location)
            #print(reply_point_of_intrest)
            msg.body(reply_point_of_intrest[:1000])
            print(msg)
            return str(response)
        elif category == "review":
            reviews = get_reviews(location)
            reply2 = json.dumps(reviews, indent=2)
            print(reply2)
            msg.body(reply2[:1000])
            print(msg)
            return str(response)
        elif category == "video":
            video_urls = get_videos(location)
            reply3 = "Videos:\n" + "\n".join(video_urls)
            print(reply3)
            msg.body(reply3[:1000])
            print(msg)
            return str(response)
        else:
            wiki_info = get_wikipedia_info(location)
            if wiki_info:
                general_info = f"{wiki_info['title']}\n{wiki_info['summary'][:100]}"                
            else:
                general_info = get_general_info(location)
                reply = json.dumps(general_info, indent=2)
            reply_point_of_intrest = search_and_populate_poi_text(location)
            #print(reply_point_of_intrest[:1000])
            msg.body(general_info +'\n'+reply_point_of_intrest[:1000])
            return str(response)
    except Exception as e:
        reply = str(e)

    msg.body(reply[:1000])
    return str(response)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
