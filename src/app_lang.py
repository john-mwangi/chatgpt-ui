"""Calls OpenAI API via LangChain"""

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from src import utils
from src.params import models
from src.prompt_lang import memory, prompt_template

load_dotenv()


def display_cost(tokens, prompt_cost, conv_cost):
    return f"""
    <p style="font-size:12px;text-align:right;">
    Tokens used: {tokens}<br>
    Prompt cost: {prompt_cost}<br>
    Conversation cost: {conv_cost}
    </p>
    """


def chat_history():
    # Chat history
    msgs_list = memory.chat_memory.dict()["messages"]
    messages = [(m["type"].upper(), m["content"]) for m in msgs_list]

    for msg in messages:
        role = msg[0]
        content = msg[1]

        with st.chat_message(name=role):
            st.markdown(content)


def handle_prompt(model: str):
    # Prompt handling
    prompt = st.chat_input(placeholder="Say something...")

    if prompt:
        with st.chat_message(name="user"):
            st.markdown(prompt)

        # Handle response
        llm_chain = LLMChain(
            llm=ChatOpenAI(model=model), prompt=prompt_template, memory=memory
        )

        response = llm_chain.predict(question=prompt)

        with st.chat_message(name="assistant"):
            st.markdown(response)

        costs = utils.calculate_cost(
            prompt=prompt, model=model, response=response
        )

        st.write(
            display_cost(
                tokens=costs["tokens_used"],
                prompt_cost=costs["prompt_cost"],
                conv_cost=costs["conversation_cost"],
            ),
            unsafe_allow_html=True,
        )
