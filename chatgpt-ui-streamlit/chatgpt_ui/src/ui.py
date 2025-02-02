"""Calls OpenAI API via LangChain"""

import os

import git
import streamlit as st
from git.exc import InvalidGitRepositoryError

from chatgpt_ui.configs import PKG_DIR, template
from chatgpt_ui.configs.params import Settings
from chatgpt_ui.utils import auth
from chatgpt_ui.utils.utils import clear_conversation

try:
    repo = git.Repo(PKG_DIR.parent)
except InvalidGitRepositoryError as e:
    repo = git.Repo(os.getcwd())

main = repo.head.reference
latest_commit = main.commit.hexsha


def display_cost(**kwargs):
    logprobs_text = (
        f"Log Probabilities: {kwargs['logprobs']:.5f}"
        if kwargs.get("logprobs") is not None
        else ""
    )

    return f"""
    <p style="font-size:12px;text-align:right;">
    Model: {kwargs["model"]}<br>
    Tokens used: {kwargs["tokens"]}<br>
    Prompt cost: {kwargs["prompt_cost"]:.5f}<br>
    Conversation cost: {kwargs["conv_cost"]:.5f}<br>
    {logprobs_text}
    </p>
    """


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

    st.sidebar.text(f"latest commit: {latest_commit[:7]}")

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

        st.write(f"Prompt Template: {template}")

        user_info = st.text_area(
            label="User Info",
            value=st.session_state.user_info,
            help="This role will give context to the GPT app",
            placeholder="Enter a role for the Chat GPT app",
            disabled=True,
        )

    return model
