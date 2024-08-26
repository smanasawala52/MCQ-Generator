from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
import io
from PIL import Image
import requests
from StreamlitImageReaderHuggingFace import huggingface_load_image_extractor_flask

app = Flask(__name__)


@app.route('/whatsapp', methods=['GET', 'POST'])
def whatsapp_webhook():
    print('Received a message')
    media_url = request.form.get('MediaUrl0')
    if media_url:
        response = requests.get(media_url)
        image = Image.open(io.BytesIO(response.content))
        search_result_url = huggingface_load_image_extractor_flask(image)  # Adapt this function to return URL

        resp = MessagingResponse()
        resp.message(f'Here is the search result: {search_result_url}')
        return str(resp)
    else:
        resp = MessagingResponse()
        resp.message("Please send an image for search.")
        return str(resp)


if __name__ == '__main__':
    app.run(debug=True)
