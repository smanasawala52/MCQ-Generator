import os
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

logging.info("Hi! Reached here.")
#importing necessary packages from langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

load_dotenv()
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o", temperature=0.7)
logging.info(llm)

def generate_iteninary_single_prompts():
    quiz_generation_template = """
    Take a deep breath and work on this step by step.
    packages details: {text}
    Most importantly, make use of time_of_day attribute to make sure trips are organized correctly.
    You are an expert Iteninary maker. Given the above packages details, it is your job to \
    create a iteninary for {number} of days for customer looking for {subject} holidays on {tone}. \
    Make sure days are not repeated and check all the days to be confirming the text as well. \
    Make sure if user wants to explore multiple cities, then first give iteninary for one city and then for other. \
    Include check-in and check-out details for each city, ensuring a smooth transition between cities. \
    Start Trip with check-in and Keep last day for check-out only. \
    Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
    Ensure to make {number} days.
    {response_json}
    """
    quiz_generation_prompt = PromptTemplate(
        input_variables=["text", "number", "subject", "tone", "response_json"],
        template=quiz_generation_template
    )
    quiz_generation_chain=LLMChain(llm=llm,prompt=quiz_generation_prompt,output_key="quiz",verbose=True)
    return quiz_generation_chain

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
    quiz_generation_chain=LLMChain(llm=llm,prompt=quiz_generation_prompt,output_key="quiz",verbose=True)
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
    poi_generation_chain=LLMChain(llm=llm,prompt=poi_generation_prompt,output_key="point_of_intrests",verbose=True)
    return poi_generation_chain