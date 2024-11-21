import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from streamlit.runtime.scriptrunner.script_run_context import (
    get_script_run_ctx,
)

from chatgpt_ui.configs.params import Settings
from chatgpt_ui.src.langchain import prompt_template
from chatgpt_ui.src.ui import create_ui, display_cost
from chatgpt_ui.utils.utils import CalculateCosts, calc_logprobs

ctx = get_script_run_ctx()
session_id = ctx.session_id

if "store" not in st.session_state:
    st.session_state.store = {}

store = st.session_state.store


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def create_app():
    """The LangChain based app"""

    # Main UI
    model = create_ui()

    # LangChain functionality
    # Chat history
    memory = store.get(session_id)
    if memory:
        msgs_list = memory.dict()["messages"]
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
            llm = ChatOpenAI(model=model, **Settings.load().gpt_params)
        elif model.startswith("claude"):
            llm = ChatAnthropic(model=model)
        else:
            st.error("Unknown model")

        runnable = prompt_template | llm

        with_message_history = RunnableWithMessageHistory(
            runnable=runnable,
            get_session_history=get_session_history,
        )

        ai_message = with_message_history.invoke(
            {"question": prompt},
            config={"configurable": {"session_id": session_id}},
        )

        response = ai_message.content

        with st.chat_message(name="assistant"):
            st.markdown(response)

        calc = CalculateCosts()
        costs = calc.calculate_cost(
            prompt=prompt, model=model, response=response
        )

        logprobs = ai_message.response_metadata.get("logprobs")
        log_probs = calc_logprobs(logprobs)

        st.write(
            display_cost(
                **dict(
                    model=costs["model"],
                    tokens=costs["tokens_used"],
                    prompt_cost=costs["prompt_cost"],
                    conv_cost=costs["conversation_cost"],
                    logprobs=log_probs,
                )
            ),
            unsafe_allow_html=True,
        )
