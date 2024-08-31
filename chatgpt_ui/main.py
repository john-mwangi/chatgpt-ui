import streamlit as st

from chatgpt_ui.src.use_langchain import run_app
from chatgpt_ui.utils.registration import authenticate_user

if "user_info" not in st.session_state:
    authenticate_user(allow_new_users=False)

else:
    run_app()
