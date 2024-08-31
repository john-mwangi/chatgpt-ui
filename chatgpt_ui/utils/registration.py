import streamlit as st

from chatgpt_ui.utils import auth


def authenticate_user(allow_new_users: bool = True):
    col1, col2, col3 = st.columns([1, 2, 1])

    user_opts = ("Yes", "No", "I forgot my password")
    if not allow_new_users:
        user_opts = ("Yes", "I forgot my password")

    # Authentication form layout
    do_you_have_an_account = col2.selectbox(
        label="Do you have an account?",
        options=user_opts,
    )
    auth_form = col2.form(key="Authentication form", clear_on_submit=False)
    email = auth_form.text_input(label="Email")
    password = (
        auth_form.text_input(label="Password", type="password")
        if do_you_have_an_account in {"Yes", "No"}
        else auth_form.empty()
    )
    auth_notification = col2.empty()

    # Sign In
    if do_you_have_an_account == "Yes" and auth_form.form_submit_button(
        label="Sign In", use_container_width=True, type="primary"
    ):
        with auth_notification, st.spinner("Signing in"):
            auth.sign_in(email, password)

    # Create Account
    elif do_you_have_an_account == "No" and auth_form.form_submit_button(
        label="Create Account", use_container_width=True, type="primary"
    ):
        with auth_notification, st.spinner("Creating account"):
            auth.create_account(email, password)

    # Password Reset
    elif (
        do_you_have_an_account == "I forgot my password"
        and auth_form.form_submit_button(
            label="Send Password Reset Email",
            use_container_width=True,
            type="primary",
        )
    ):
        with auth_notification, st.spinner("Sending password reset link"):
            auth.reset_password(email)

    # Authentication success and warning messages
    if "auth_success" in st.session_state:
        auth_notification.success(st.session_state.auth_success)
        del st.session_state.auth_success
    elif "auth_warning" in st.session_state:
        auth_notification.warning(st.session_state.auth_warning)
        del st.session_state.auth_warning