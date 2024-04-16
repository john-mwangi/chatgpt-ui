"""Calls OpenAI API via LangChain"""

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from src.params import models
from src.prompt_lang import memory, prompt_template

load_dotenv()

# Side bar
st.sidebar.title("ChatGPT API Interface")
model = st.sidebar.selectbox(label="Select a model", options=models)

# Chat history
msgs_list = memory.chat_memory.dict()["messages"]
messages = [(m["type"].upper(), m["content"]) for m in msgs_list]

for msg in messages:
    role = msg[0]
    content = msg[1]

    with st.chat_message(name=role):
        st.markdown(content)

# Prompt handling
prompt = st.chat_input(placeholder="Say something...")
if prompt:

    with st.chat_message(name="user"):
        st.markdown(prompt)

    llm_chain = LLMChain(
        llm=ChatOpenAI(model=model), prompt=prompt_template, memory=memory
    )

    # Handle response
    response = llm_chain.predict(question=prompt)

    with st.chat_message(name="assistant"):
        st.markdown(response)
