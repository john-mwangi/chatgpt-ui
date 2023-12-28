"""Calls OpenAI API directly"""

import os
import pickle

import openai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

CHATGPT_ROLE = "You are an expert Python programmer"
API_KEY = os.environ["API_KEY"]
openai.api_key = API_KEY

model_pricing = {
    "gpt-4-1106-preview": {
        "input_cost_usd_per_1K_tokens": 0.01,
        "output_cost_usd_per_1K_tokens": 0.03,
    },
    "gpt-3.5-turbo-1106": {
        "input_cost_usd_per_1K_tokens": 0.0010,
        "output_cost_usd_per_1K_tokens": 0.0020,
    },
}


def prompt_gpt(
    model: str = None,
    messages: list[str] = None,
) -> dict:
    """
    Submit a prompt to ChatGPT API including a conversation history.

    Args:
    ---
    model: The name of the model
    messages: A list of messages the includes the prompt and the conversation history

    Returns:
    ---
    A dictionary that has the following keys:
    - msg: ChatGPT response
    - msgs: The updated conversation history
    - input_tokens: Number of input tokens used
    - output_tokens: Number of output tokens used

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
        "msgs": messages,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def calculate_cost(input_tokens: int, output_tokens: int, model: str):
    """Calculates the cost of the prompt."""

    input_cost_usd_per_1K_tokens = model_pricing.get(model).get(
        "input_cost_usd_per_1K_tokens"
    )
    output_cost_usd_per_1K_tokens = model_pricing.get(model).get(
        "output_cost_usd_per_1K_tokens"
    )

    input_tokens_thousands = input_tokens / 1000
    output_tokens_thousands = output_tokens / 1000

    input_cost = input_tokens_thousands * input_cost_usd_per_1K_tokens
    output_cost = output_tokens_thousands * output_cost_usd_per_1K_tokens

    token_used = input_tokens + output_tokens
    promt_cost = input_cost + output_cost

    return token_used, promt_cost


def create_messages(
    prompt: str = None,
    role: str = CHATGPT_ROLE,
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


def calc_conversation_cost(prompt_cost: float, new_conversation: bool) -> float:
    prev_costs = [0]

    if not new_conversation:
        try:
            with open("costs.pkl", mode="rb") as f:
                prev_costs: list[float] = pickle.load(f)
        except Exception as e:
            print(e)

    prev_costs.append(prompt_cost)

    with open("costs.pkl", mode="wb") as f:
        pickle.dump(prev_costs, file=f)

    return sum(prev_costs)


# App interface, capture prompt
st.sidebar.title("ChatGPT API Interface")
model = st.sidebar.selectbox(
    label="Select a model", options=["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]
)
new_conversation = st.sidebar.checkbox(label="Start new conversation?", value=True)
prompt = st.sidebar.text_area(
    label="Prompt", placeholder="Enter your prompt here...", height=250
)
submit = st.sidebar.button(label="Submit")

# Load conversation history
conversation_history = None

if not new_conversation:
    if not os.path.exists("messages.pkl"):
        st.text("No history to load")
    else:
        try:
            with open("messages.pkl", mode="rb") as f:
                conversation_history = pickle.load(f)
        except Exception as e:
            print(e)

# Create messages
messages = create_messages(prompt, messages=conversation_history)

# Process submission
if submit:
    with st.spinner():
        result = prompt_gpt(model=model, messages=messages)
    token_used, promt_cost = calculate_cost(
        input_tokens=result.get("input_tokens"),
        output_tokens=result.get("output_tokens"),
        model=model,
    )

    conversation_cost = calc_conversation_cost(
        prompt_cost=promt_cost, new_conversation=new_conversation
    )

    st.text(f"Tokens used: {token_used}")
    st.text(f"Prompt cost USD: {promt_cost}")
    st.text(f"Conversation cost USD: {conversation_cost}")

    st.markdown(result.get("msg"))

    with open("messages.pkl", mode="wb") as f:
        pickle.dump(result["msgs"], file=f)

# TODO:save different conversations in different files
