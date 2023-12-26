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


def create_messages(
    prompt: str = None,
    role: str = "You are an expert Python programmer",
    messages: list[dict] = None,
):
    """Adds the user prompt to the conversation history."""

    if prompt is None:
        raise ValueError("prompt cannot be None")

    if messages is None:
        messages = [
            {"role": "system", "content": role},
            {"role": "user", "content": prompt},
        ]
    else:
        messages.append({"role": "user", "content": prompt})

    return messages


# App interface, capture prompt
st.title("ChatGPT API Interface")
model = st.selectbox(label="Model", options=["gpt-4-1106-preview", "gpt-3.5-turbo"])
new_conversation = st.checkbox(label="Start new conversation?", value=False)
prompt = st.text_area(
    label="Prompt", placeholder="Enter your prompt here...", height=100
)

# Load conversation history
conversation_history = None

if not new_conversation:
    if not os.path.exists("messages.pkl"):
        st.text("No history to load")
    else:
        try:
            with open("messages.pkl", mode="rb") as f:
                conversation_history = pickle.load(f)
        except Exception:
            pass

# Create messages
messages = create_messages(prompt, messages=conversation_history)

# Process submission
submit = st.button(label="Submit")

if submit:
    result = prompt_gpt(model=model, messages=messages)
    token_used, promt_cost = calculate_cost(
        input_tokens=result.get("input_tokens"),
        output_tokens=result.get("output_tokens"),
    )

    st.text(f"Tokens used: {token_used}")
    st.text(f"Prompt cost USD: {promt_cost}")
    st.markdown(result.get("msg"))

    with open("messages.pkl", mode="wb") as f:
        pickle.dump(result["messages"], file=f)
