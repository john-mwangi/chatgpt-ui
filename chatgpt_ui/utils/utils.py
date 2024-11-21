import pickle

import numpy as np
import streamlit as st
import tiktoken
from stop_words import get_stop_words

from chatgpt_ui.configs import costs_path
from chatgpt_ui.configs.params import Settings


class CalculateCosts:
    """Class for calculating the costs of conversations with LLMs"""

    def calc_prompt_cost(
        self, input_tokens: int, output_tokens: int, model: str
    ):
        """Calculates the cost of the prompt."""

        model_pricing = Settings.load().pricing

        input_price = model_pricing.get(model).get("input_price")
        output_price = model_pricing.get(model).get("output_price")

        input_tokens_thousands = input_tokens / 1000
        output_tokens_thousands = output_tokens / 1000

        input_cost = input_tokens_thousands * input_price
        output_cost = output_tokens_thousands * output_price

        token_used = input_tokens + output_tokens
        promt_cost = input_cost + output_cost

        return token_used, promt_cost

    def calc_conversation_cost(
        self, prompt_cost: float, new_conversation: bool
    ) -> float:
        prev_costs = [0]

        if not new_conversation:
            try:
                with open(costs_path, mode="rb") as f:
                    prev_costs: list[float] = pickle.load(f)
            except Exception as e:
                print(e)

        prev_costs.append(prompt_cost)

        if not costs_path.exists():
            costs_path.parent.mkdir()

        with open(costs_path, mode="wb") as f:
            pickle.dump(prev_costs, file=f)

        return sum(prev_costs)

    def num_tokens_from_string(self, message: str, model: str):
        """Count the number of tokes from a string"""

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(message)

        return len(tokens)

    def calculate_cost(self, prompt, model, response):

        if "conversation_cost" not in st.session_state:
            st.session_state["conversation_cost"] = 0

        input_tokens = self.num_tokens_from_string(message=prompt, model=model)
        output_tokens = self.num_tokens_from_string(
            message=response, model=model
        )

        tokens_used, prompt_cost = self.calc_prompt_cost(
            input_tokens, output_tokens, model
        )

        st.session_state["conversation_cost"] += prompt_cost

        costs = {
            "model": model,
            "tokens_used": tokens_used,
            "prompt_cost": prompt_cost,
            "conversation_cost": st.session_state["conversation_cost"],
        }

        return costs


def calc_logprobs(logprobs: list[dict] | None):
    """Sums the log probabilities of the response

    Args:
    ---
    logprobs: the logprobs from the Open AI model

    Returns:
    ---
    The mean of log probabilities of non-stop words
    """
    stop_words = get_stop_words("en")

    try:
        content = logprobs["content"]
        res = np.mean(
            [
                np.exp(x["logprob"])
                for x in content
                if x["token"] not in stop_words
            ]
        )
        return res

    except Exception as e:
        print(e)
        return None


def clear_conversation():
    if "store" not in st.session_state:
        pass
    else:
        session_ids = st.session_state.store.keys()
        if session_ids:
            for id in session_ids:
                st.session_state.store[id].clear()
    st.session_state.pop("conversation_cost", default=None)


if __name__ == "__main__":
    import json

    from chatgpt_ui.configs.params import Settings

    models = Settings.load().models
    costs = CalculateCosts()

    print(
        costs.num_tokens_from_string(
            message="tiktoken is great!", model=models[0]
        )
    )

    with open("../samples/response.json", "r") as f:
        response = json.load(f)

    msg = response["choices"][0]["logprobs"]
    res = calc_logprobs(msg)
    print(res)
