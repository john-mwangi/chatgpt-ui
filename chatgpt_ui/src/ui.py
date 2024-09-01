"""Calls OpenAI API via LangChain"""

import streamlit as st

from chatgpt_ui.configs import GPT_ROLE
from chatgpt_ui.configs.params import Settings
from chatgpt_ui.src.langchain import memory
from chatgpt_ui.utils import auth


def display_cost(model, tokens, prompt_cost, conv_cost):
    return f"""
    <p style="font-size:12px;text-align:right;">
    Model: {model}<br>
    Tokens used: {tokens}<br>
    Prompt cost: {prompt_cost:.5f}<br>
    Conversation cost: {conv_cost:.5f}
    </p>
    """


def clear_conversation():
    memory.clear()
    st.session_state.pop("conversation_cost", default=None)


def create_ui():
    """Defines the UI of the app"""

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

    with st.expander(label="Manage"):
        password = st.text_input(
            label="Confirm your password", type="password"
        )

        st.button(
            label="Delete Account",
            on_click=auth.delete_account,
            args=[password],
            type="primary",
        )

        # st.write(st.session_state.user_info)

        gpt_role = st.text_area(
            label="Chat GPT role",
            value=GPT_ROLE,
            help="This role will give context to the GPT app",
            placeholder="Enter a role for the Chat GPT app",
            disabled=True,
        )

    return model
