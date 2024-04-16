import pickle

import streamlit as st
import tiktoken

from .params import costs_path, model_pricing


def calc_prompt_cost(input_tokens: int, output_tokens: int, model: str):
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


def calc_conversation_cost(
    prompt_cost: float, new_conversation: bool
) -> float:
    prev_costs = [0]

    if not new_conversation:
        try:
            with open("costs.pkl", mode="rb") as f:
                prev_costs: list[float] = pickle.load(f)
        except Exception as e:
            print(e)

    prev_costs.append(prompt_cost)

    if not costs_path.exists():
        costs_path.parent.mkdir()

    with open(costs_path, mode="wb") as f:
        pickle.dump(prev_costs, file=f)

    return sum(prev_costs)


def num_tokens_from_string(message: str, model: str):
    """Count the number of tokes from a string"""

    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(message)
    return len(tokens)


def calculate_cost(prompt, model, response):

    if "conversation_cost" not in st.session_state:
        st.session_state["conversation_cost"] = 0

    input_tokens = num_tokens_from_string(message=prompt, model=model)
    output_tokens = num_tokens_from_string(message=response, model=model)

    tokens_used, prompt_cost = calc_prompt_cost(
        input_tokens, output_tokens, model
    )

    st.session_state["conversation_cost"] += prompt_cost

    costs = {
        "tokens_used": tokens_used,
        "prompt_cost": prompt_cost,
        "conversation_cost": st.session_state["conversation_cost"],
    }

    return costs


if __name__ == "__main__":
    from params import models

    print(
        num_tokens_from_string(message="tiktoken is great!", model=models[0])
    )
