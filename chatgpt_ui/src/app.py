import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from chatgpt_ui.configs.params import Settings
from chatgpt_ui.src.langchain import memory
from chatgpt_ui.src.ui import create_ui, display_cost
from chatgpt_ui.utils.utils import CalculateCosts, calc_logprobs


def create_app():
    """The LangChain based app"""

    # Main UI
    model = create_ui()

    # LangChain functionality
    # Chat history
    msgs_list = memory.chat_memory.dict()["messages"]
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
            llm = (ChatAnthropic(model=model),)
        else:
            st.error("Unknown model")

        ai_msg = llm.invoke(("human", prompt))
        response = ai_msg.content
        logprobs = ai_msg.response_metadata["logprobs"]
        memory.save_context(
            outputs={"output": response}, inputs={"input": prompt}
        )

        log_probs = calc_logprobs(logprobs)

        with st.chat_message(name="assistant"):
            st.markdown(response)

        calc = CalculateCosts()
        costs = calc.calculate_cost(
            prompt=prompt, model=model, response=response
        )

        st.write(
            display_cost(
                dict(
                    model=costs["model"],
                    tokens=costs["tokens_used"],
                    prompt_cost=costs["prompt_cost"],
                    conv_cost=costs["conversation_cost"],
                    logprobs=log_probs,
                )
            ),
            unsafe_allow_html=True,
        )
