import os
import pickle

import openai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["API_KEY"]
openai.api_key = API_KEY


def prompt_gpt(
    model: str = None,
    messages: list[str] = None,
) -> str:
    """
    Useful links:
    ---
    models: https://platform.openai.com/docs/models/gpt-3-5
    account balance: https://platform.openai.com/account/billing/overview
    create params: https://platform.openai.com/docs/api-reference/chat/create
    pricing: https://openai.com/pricing
    """

    response = openai.ChatCompletion.create(model=model, messages=messages)
    msg = response.get("choices")[0].get("message").get("content")

    assistant = {"role": "assistant", "content": msg}
    messages.append(assistant)

    input_tokens = response.get("usage").get("prompt_tokens")
    output_tokens = response.get("usage").get("prompt_tokens")

    return {
        "msg": msg,
        "messages": messages,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def calculate_cost(input_tokens: int, output_tokens: int):
    input_cost_usd_per_1K_tokens = 0.01
    output_cost_usd_per_1K_tokens = 0.03

    input_tokens_thousands = input_tokens / 1000
    output_tokens_thousands = output_tokens / 1000

    input_cost = input_tokens_thousands * input_cost_usd_per_1K_tokens
    output_cost = output_tokens_thousands * output_cost_usd_per_1K_tokens

    token_used = input_tokens + output_tokens
    promt_cost = input_cost + output_cost

    return token_used, promt_cost


st.title("ChatGPT API Interface")
model = st.selectbox(label="Model", options=["gpt-4-1106-preview", "gpt-3.5-turbo"])
new_conversation = st.checkbox(label="Start new conversation?", value=False)
messages = st.file_uploader(label="Conversation history file", type="pkl")
prompt = st.text_area(
    label="Prompt", placeholder="Enter your prompt here...", height=100
)

role = "You are an expert Python programmer"
messages = [
    {"role": "system", "content": role},
    {"role": "user", "content": prompt},
]

submit = st.button(label="Submit")
if submit:
    if not new_conversation:
        if not os.path.exists("messages.pkl"):
            print("No history to load")
        else:
            messages.append({"role": "user", "content": prompt})

    result = prompt_gpt(model=model, messages=messages)
    token_used, promt_cost = calculate_cost(
        input_tokens=result.get("input_tokens"),
        output_tokens=result.get("output_tokens"),
    )

    st.text(f"Tokens used: {token_used}")
    st.text(f"Prompt cost USD: {promt_cost}")
    st.markdown(result.get("msg"))

    with open("messages.pkl", mode="wb") as f:
        pickle.dump(result.get("msgs"), file=f)
