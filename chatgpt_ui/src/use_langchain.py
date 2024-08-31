"""Calls OpenAI API via LangChain"""

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from chatgpt_ui.configs.params import Settings
from chatgpt_ui.src.prompt_langchain import memory, prompt_template
from chatgpt_ui.utils import auth, utils

load_dotenv()


def display_cost(tokens, prompt_cost, conv_cost):
    return f"""
    <p style="font-size:12px;text-align:right;">
    Tokens used: {tokens}<br>
    Prompt cost: {prompt_cost}<br>
    Conversation cost: {conv_cost}
    </p>
    """


def clear_conversation():
    memory.clear()
    st.session_state.pop("conversation_cost", default=None)


def run_app():
    # Side bar
    models = Settings.load().models
    st.sidebar.title("ChatGPT API Interface")
    model = st.sidebar.selectbox(label="Select a model", options=models)

    st.sidebar.divider()

    col1, col2 = st.sidebar.columns(2)

    with col1:
        st.button(
            label="Sign Out",
            on_click=auth.sign_out,
            type="primary",
            help="Log out",
        )

    with col2:
        st.button(
            label="Clear",
            on_click=clear_conversation,
            type="secondary",
            help="Clear chat conversation",
        )

    with st.expander(label="Delete your account"):
        password = st.text_input(
            label="Confirm your password", type="password"
        )

        st.button(
            label="Delete Account",
            on_click=auth.delete_account,
            args=[password],
            type="primary",
        )

        st.write(st.session_state.user_info)

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
        if model.startswith("gpt"):
            llm_chain = LLMChain(
                llm=ChatOpenAI(model=model),
                prompt=prompt_template,
                memory=memory,
            )
        elif model.startswith("claude"):
            llm_chain = LLMChain(
                llm=ChatAnthropic(
                    model="claude-3-5-sonnet-20240620",
                    # temperature=0,
                    # max_tokens=1024,
                    # timeout=None,
                    # max_retries=2,
                ),
                prompt=prompt_template,
                memory=memory,
            )
        else:
            raise ValueError("Unknown model")

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
