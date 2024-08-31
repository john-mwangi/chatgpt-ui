import streamlit as st

from chatgpt_ui.src.use_langchain import app
from chatgpt_ui.utils.registration import authenticate_user

if "user_info" not in st.session_state:
    authenticate_user(allow_new_users=False)

else:
    app()
