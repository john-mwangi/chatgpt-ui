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
    Tokens: {tokens}<br>
    Prompt cost: {prompt_cost}<br>
    Conversation cost: {conv_cost}
    </p>
    """


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

    # Handle response
    llm_chain = LLMChain(
        llm=ChatOpenAI(model=model), prompt=prompt_template, memory=memory
    )

    response = llm_chain.predict(question=prompt)

    with st.chat_message(name="assistant"):
        st.markdown(response)

    # Cost calculation
    input_tokens = utils.num_tokens_from_string(message=prompt, model=model)
    output_tokens = utils.num_tokens_from_string(message=response, model=model)

    token_used, prompt_cost = utils.calc_prompt_cost(
        input_tokens, output_tokens, model
    )
    conversation_cost = utils.calc_conversation_cost(
        prompt_cost=prompt_cost, new_conversation=True
    )

    result = {}
    result["msg"] = response
    result["token_used"] = token_used
    result["prompt_cost"] = prompt_cost
    result["conversation_cost"] = conversation_cost

    token_used = result.get("token_used")
    prompt_cost = result.get("prompt_cost")
    conversation_cost = result.get("conversation_cost")

    st.write(
        display_cost(
            tokens=token_used,
            prompt_cost=prompt_cost,
            conv_cost=conversation_cost,
        ),
        unsafe_allow_html=True,
    )
