"""Calls OpenAI API via LangChain"""

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from src.params import models
from src.prompt_lang import memory, prompt_template
from src.utils import (
    calc_conversation_cost,
    calc_prompt_cost,
    num_tokens_from_string,
)

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

    # Cost calculation
    input_tokens = num_tokens_from_string(message=prompt, model=model)
    output_tokens = num_tokens_from_string(message=response, model=model)

    token_used, promt_cost = calc_prompt_cost(
        input_tokens, output_tokens, model
    )
    conversation_cost = calc_conversation_cost(
        prompt_cost=promt_cost, new_conversation=True
    )

    result = {}
    result["msg"] = response
    result["token_used"] = token_used
    result["promt_cost"] = promt_cost
    result["conversation_cost"] = conversation_cost

    token_used = result.get("token_used")
    promt_cost = result.get("promt_cost")
    conversation_cost = result.get("conversation_cost")

    st.text(f"Tokens used: {token_used}")
    st.text(f"Prompt cost USD: {promt_cost}")
    st.text(f"Conversation cost USD: {conversation_cost}")
