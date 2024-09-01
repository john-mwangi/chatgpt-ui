import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from chatgpt_ui.src.prompt_langchain import memory, prompt_template
from chatgpt_ui.src.ui import display_cost, ui
from chatgpt_ui.utils.utils import CalculateCosts


def app():
    """The LangChain based app"""

    # Main UI
    model = ui()

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
            llm_chain = LLMChain(
                llm=ChatOpenAI(model=model),
                prompt=prompt_template,
                memory=memory,
            )
        elif model.startswith("claude"):
            llm_chain = LLMChain(
                llm=ChatAnthropic(model=model),
                prompt=prompt_template,
                memory=memory,
            )
        else:
            st.error("Unknown model")

        response = llm_chain.predict(question=prompt)

        with st.chat_message(name="assistant"):
            st.markdown(response)

        calc = CalculateCosts()
        costs = calc.calculate_cost(
            prompt=prompt, model=model, response=response
        )

        st.write(
            display_cost(
                model=costs["model"],
                tokens=costs["tokens_used"],
                prompt_cost=costs["prompt_cost"],
                conv_cost=costs["conversation_cost"],
            ),
            unsafe_allow_html=True,
        )
