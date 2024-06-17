#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data_iteninary,display_Iteninary_gpt, open_file_in_same_directory
from src.mcqgenerator.logger import logging

from src.mcqgenerator.IteninaryGenerator import generate_iteninary_prompts
from src.mcqgenerator.IteninaryGeneratorSingle import generate_iteninary_single_prompts

import streamlit as st
from langchain.callbacks import get_openai_callback




def generate_home_page(verticle):
    st.title="MCQ's App"
    with st.form('user_inputs'):
        # file upload
        uploaded_file = st.file_uploader('Upload a PDF or txt file.')
        # input_fields
        mcq_count = st.number_input('No. of Days', min_value=1,
                                    max_value=50)
        mcq_subject = st.text_input('Insert City', max_chars=50)
        # Define the options for the drop-down menu
        complexity_options = ['Mountain', 'Beach', 'Wildlife']
        # Create a drop-down menu
        mcq_tone = st.selectbox('Places of Intrests',
                                complexity_options, index=0,
                                help='Select the places of intrests'
                                )
        # Display the selected complexity level
        st.write('You selected:', mcq_tone)
        mcq_submit_button = st.form_submit_button('Create Iteninary')
        if mcq_submit_button and uploaded_file is not None \
            and mcq_count and mcq_subject and mcq_tone:
            with st.spinner('loading.....'):
                try:
                    quiz_generation_template_response_json = \
                        open_file_in_same_directory('response_jsons\\IteninaryResponse.json'
                            )
                    quiz_data = read_file(uploaded_file)
                    print(quiz_data)
                    with get_openai_callback() as cb:
                        generate_iteninary_prompts_temp=generate_iteninary_single_prompts()
                        result = generate_iteninary_prompts_temp({
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
                        print('--start result-')
                        print(result)
                        print('--end result-')
                        quiz1=result.get("quiz",None).replace("json", "").replace("```","").replace("```","")

                        print('--start quiz1-')
                        print(quiz1)
                        print('--end quiz1-')
                        if quiz1 is not None:
                            quiz=json.loads(quiz1)
                            print('--start quiz-')
                            print(quiz)
                            print('--end quiz-')
                            display_Iteninary_gpt(st,quiz)
                            quiz_table_data=get_table_data_iteninary(quiz)
                            if quiz_table_data is not None:
                                df=pd.DataFrame(quiz_table_data)
                                df.index = df.index+1
                                #st.table(df)
                                #st.text_area(label="Review", value=result["review"])
                                #st.write(result["review"])
                            else:
                                st.error("Error in table data")
                    else:
                        st.write(result)

generate_home_page("MCQ")