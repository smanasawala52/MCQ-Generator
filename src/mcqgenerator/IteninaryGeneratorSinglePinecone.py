import json
import os
import pandas as pd
import pinecone
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback

from src.mcqgenerator.IteninaryGeneratorSingle import generate_iteninary_single_prompts
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

logging.info("Hi! Reached here.")
# importing necessary packages from langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import Pinecone as PineconeStore
from langchain_openai import OpenAIEmbeddings

# Define metadata field information
metadata_field_info = [
    AttributeInfo(
        name="city",
        description="The name of the city",
        type="string",
    ),
    AttributeInfo(
        name="about",
        description="A detailed description of the city and its attractions",
        type="string",
    ),
    AttributeInfo(
        name="packages",
        description="A list of travel packages available for the city",
        type="list[object]",
    ),
    AttributeInfo(
        name="time_of_day",
        description="The time of day when the package activity takes place",
        type="string",
    ),
    AttributeInfo(
        name="popularity_score",
        description="The popularity score of the package activity",
        type="float",
    ),
    AttributeInfo(
        name="adventure_score",
        description="The adventure score of the package activity",
        type="float",
    ),
    AttributeInfo(
        name="activity_type",
        description="The type of activity in the package",
        type="string",
    ),
    AttributeInfo(
        name="activity",
        description="A brief description of the activity in the package",
        type="string",
    ),
    AttributeInfo(
        name="details",
        description="Detailed information about the package activity",
        type="string",
    ),
    AttributeInfo(
        name="image_url",
        description="A list of URLs for images related to the package",
        type="list[string]",
    ),
    AttributeInfo(
        name="places_covered",
        description="A list of places covered by the package",
        type="list[object]",
    ),
    AttributeInfo(
        name="place_covered_name",
        description="The name of the place covered by the package",
        type="string",
    ),
    AttributeInfo(
        name="place_covered_image_url",
        description="A list of URLs for images of the place covered by the package",
        type="list[string]",
    ),
]

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
KT_PINECONE_API_KEY = os.getenv('KT_PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = KT_PINECONE_API_KEY
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
index_name = "travel-project"
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o", temperature=0.7)
logging.info(llm)
# vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding)
pc = pinecone.Pinecone(
    api_key=KT_PINECONE_API_KEY  # Replace with your actual Pinecone API key
)
index = pc.Index(index_name)


def generate_itinerary_with_pinecone(number_of_days, city, tone, quiz_generation_template_response_json):
    # Query Pinecone for relevant activities
    quiz_generation_template = f"""
            Take a deep breath and work on this step by step.
            Most importantly, make use of time_of_day attribute to make sure trips are organized correctly.
            You are an expert Iteninary maker. Given the above packages details, it is your job to \
            create a iteninary for {number_of_days} of days for customer looking for {city} holidays on {tone}. \
            Make sure days are not repeated and check all the days to be confirming the text as well. \
            Make sure if user wants to explore multiple cities, then first give iteninary for one city and then for other. \
            Include check-in and check-out details for each city, ensuring a smooth transition between cities. \
            Start Trip with check-in and Keep last day for check-out only. \
            Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
            Ensure to make {number_of_days} days.
            {quiz_generation_template_response_json}
            """
    # query_text = f"activities in {city}"
    query_vector = embeddings.embed_query(quiz_generation_template)

    results = index.query(vector=query_vector, top_k=5, include_metadata=True)

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
        print(unique_results)
        # Integrate Pinecone results into the prompt
        return unique_results
    else:
        # st.warning("No results returned from Pinecone.")
        return None


def generate_iteninary_poi_single_prompts():
    quiz_generation_template = """
    Take a deep breath and work on this step by step.
    packages details: {text}
    Most importantly, make use of time_of_day attribute to make sure trips are organized correctly.
    You are an expert Iteninary maker. Given the above packages details, it is your job to \
    create a iteninary for selected places: {places} by customer for holidays in {city}. \
    Make sure days are not repeated and check all the days to be confirming the text as well. \
    Make sure if user wants to explore multiple cities, then first give iteninary for one city and then for other. \
    Include check-in and check-out details for each city, ensuring a smooth transition between cities. \
    Start Trip with check-in and Keep last day for check-out only. \
    Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
    Ensure to make number days.
    {response_json}
    """
    quiz_generation_prompt = PromptTemplate(
        input_variables=["text", "places", "city", "response_json"],
        template=quiz_generation_template
    )
    quiz_generation_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)
    return quiz_generation_chain


def generate_poi_single_prompts():
    poi_generation_template = """
    Take a deep breath and work on this step by step.
    read this packages details: {packages}
    Extract point of intrests for the {city} from places_covered. Return Data in Below JSON format.
    Only return point_of_intrests json only.
    Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
    {poi_response_json}
    """
    poi_generation_prompt = PromptTemplate(
        input_variables=["packages", "city", "poi_response_json"],
        template=poi_generation_template
    )
    poi_generation_chain = LLMChain(llm=llm, prompt=poi_generation_prompt, output_key="point_of_intrests", verbose=True)
    return poi_generation_chain
