import streamlit as st
from utils.user_registration import authenticate_user

from chatgpt_ui.src.use_langchain import run_app

if "user_info" not in st.session_state:
    authenticate_user(allow_new_users=False)

else:
    run_app()
