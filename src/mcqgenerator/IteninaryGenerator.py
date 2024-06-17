import os
import json
import traceback
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

def generate_iteninary_prompts():
    load_dotenv()
    OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.7)
    logging.info(llm)

    quiz_generation_template = """
    Take a deep breath and work on this step by step.
    packages details: {text}
    You are an expert Iteninary maker. Given the above packages details, it is your job to \
    Most importantly, make use of time_of_day attribute to make sure trips are organized correctly.
    create a iteninary for {number} of days for customer looking for {subject} holidays on {tone}. \
    Make sure days are not repeated and check all the days to be confirming the text as well. \
    Make sure if user wants to explore multiple cities, then first give iteninary for one city and then for other. \
    Include check-in and check-out details for each city, ensuring a smooth transition between cities. \
    Keep last day for check-out only \
    Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
    Ensure to make {number} days.
    {response_json}
    """
    quiz_generation_prompt = PromptTemplate(
        input_variables=["text", "number", "subject", "tone", "response_json"],
        template=quiz_generation_template
    )
    quiz_generation_chain=LLMChain(llm=llm,prompt=quiz_generation_prompt,output_key="quiz",verbose=True)
    quiz_generation_chain

    quiz_evaluation_template="""
    Take a deep breath and work on this step by step.
    You are an expert travel operator and planner. Given the Itinanry for {subject} holidays. \
    You need to evaluate the tone of the holidays and give complete analysis of the iteninary. \
    Only use max 50 words from the tone. \
    If the iteninary is not as per the with the tone and subject of the holidays, \
    update the iteninary which needs to be changed and \
    change the tone such that it perfectly fits the customers subject \
    Iteninary:
    {quiz}

    Check from an expert travel operator of above iteninary:
    """

    quiz_evaluation_template

    quiz_evaluation_prompt = PromptTemplate(
        input_variables=["subject", "quiz"],
        template=quiz_evaluation_template
    )
    quiz_evaluation_chain=LLMChain(
        llm=llm,
        prompt=quiz_evaluation_prompt,
        output_key="review",
        verbose=True
        )
    quiz_evaluation_chain

    generate_evaluate_chain=SequentialChain(chains=[quiz_generation_chain,quiz_evaluation_chain],
                                            input_variables=["text", "number", "subject", "tone", "response_json"],
                                            output_variables=["quiz", "review"],
                                            verbose=True)
    print(generate_evaluate_chain)
    return generate_evaluate_chain