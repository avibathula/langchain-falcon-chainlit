#!/usr/bin/env python

import os
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceEndpoint(huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
                                   repo_id=repo_id,
                                   temperature = 0.7,
                                   max_new_tokens = 700,)


template = """
You are a helpful AI assistant and provide the answer for the question asked politely.

{question}
Answer: Let's think step by step
"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = prompt | llm

question = "How to cook Pizza ? explain clearly, with ingredients"

print(llm_chain.invoke(question))
