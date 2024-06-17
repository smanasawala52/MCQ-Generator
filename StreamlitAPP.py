#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

from src.mcqgenerator.MCQGenerator import generate_evaluate_chain

import streamlit as st
from langchain.callbacks import get_openai_callback


def open_file_in_same_directory(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)

    with open(file_path, 'r') as file:
        content = file.read()

    return content


def generate_home_page(verticle):
    st.title="MCQ's App"
    with st.form('user_inputs'):
        # file upload
        uploaded_file = st.file_uploader('Upload a PDF or txt file.')
        # input_fields
        mcq_count = st.number_input('No. of MCQs', min_value=3,
                                    max_value=50)
        mcq_subject = st.text_input('Insert Subject', max_chars=20)
        # Define the options for the drop-down menu
        complexity_options = ['Simple', 'Medium', 'Complex']
        # Create a drop-down menu
        mcq_tone = st.selectbox('Complexity Level Questions',
                                complexity_options, index=0,
                                help='Select the complexity level for the questions'
                                )
        # Display the selected complexity level
        st.write('You selected:', mcq_tone)
        mcq_submit_button = st.form_submit_button('Create MCQs')
        if mcq_submit_button and uploaded_file is not None \
            and mcq_count and mcq_subject and mcq_tone:
            with st.spinner('loading.....'):
                try:
                    quiz_generation_template_response_json = \
                        open_file_in_same_directory('response_jsons\\QuizResponse.json'
                            )
                    quiz_data = read_file(uploaded_file)
                    with get_openai_callback() as cb:
                        result = generate_evaluate_chain({
                            'text': quiz_data,
                            'number': mcq_count,
                            'subject': mcq_subject,
                            'tone': mcq_tone,
                            'response_json': json.dumps(quiz_generation_template_response_json),
                            })
                        print(result)
                        print('---')
                except Exception as e:
                    traceback.print_exception(type(e),e, e.__traceback__)
                    st.error("Error")
                else:
                    print()
                    print(f"Total Tokens: {cb.total_tokens}")
                    print(f"Prompt Tokens: {cb.prompt_tokens}")
                    print(f"Completion Tokens: {cb.completion_tokens}")
                    print(f"Total Cost (USD): ${cb.total_cost}")
                    if isinstance (result, dict):
                        quiz=result.get("quiz",None)
                        if quiz is not None:
                            quiz=json.loads(quiz)
                            print('--start quiz-')
                            print(quiz)
                            print('--end quiz-')
                            quiz_table_data=get_table_data(quiz)
                            if quiz_table_data is not None:
                                df=pd.DataFrame(quiz_table_data)
                                df.index = df.index+1
                                st.table(df)
                                st.text_area(label="Review", value=result["review"])
                            else:
                                st.error("Error in table data")
                    else:
                        st.write(result)

generate_home_page("MCQ")