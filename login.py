import streamlit as st

from src import auth_functions
from src.app_lang import display_chat_history, handle_prompt
from src.params import models
from src.user_registration import authenticate_user

if "user_info" not in st.session_state:
    authenticate_user()

else:
    # Side bar
    st.sidebar.title("ChatGPT API Interface")

    model = st.sidebar.selectbox(label="Select a model", options=models)

    st.sidebar.divider()

    st.sidebar.button(
        label="Sign Out", on_click=auth_functions.sign_out, type="primary"
    )

    password = st.sidebar.text_input(
        label="Confirm your password", type="password"
    )

    st.sidebar.button(
        label="Delete Account",
        on_click=auth_functions.delete_account,
        args=[password],
        type="secondary",
    )

    display_chat_history()
    handle_prompt(model)
