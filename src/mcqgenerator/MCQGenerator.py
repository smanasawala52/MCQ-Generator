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

def generate_mcq_prompts():
    load_dotenv()
    OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.7)
    logging.info(llm)

    quiz_generation_template = """
    Take a deep breath and work on this step by step.
    Text: {text}
    You are an expert MCQ maker. Given the above text, it is your job to \
    create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. \
    Make sure questions are not repeated and check all the questions to be confirming the text as well.
    Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
    Ensure to make {number} MCQs.
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
    You are an expert english grammarian and writer. Given the Multiple Choice Quiz for {subject} students. \
    You need to evaluate the complexity of the question and give complete analysis of the quiz. \
    Only use max 50 words from the complexities. \
    If the quiz is not as per the with the cognitive and analytical abilities of the student, \
    update the quiz questions which needs to be changed and \
    change the tone such that it perfectly fits the student abilities \
    Quiz MCQs:
    {quiz}

    Check from an expert English Writer of above quiz:
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