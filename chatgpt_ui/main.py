import streamlit as st

from chatgpt_ui.src.app import create_app
from chatgpt_ui.utils.registration import authenticate_user

if "user_info" not in st.session_state:
    authenticate_user(allow_new_users=False)

else:
    create_app()
