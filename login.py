import streamlit as st

from src.app_lang import run_app
from src.user_registration import authenticate_user

if "user_info" not in st.session_state:
    authenticate_user()

else:
    run_app()
