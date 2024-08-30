from pathlib import Path

models = ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]
CHATGPT_ROLE = "You are an expert Python programmer."
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

ROOT_DIR = Path(__file__).parent.parent.resolve()
costs_path = ROOT_DIR / "files/costs.pkl"
msgs_path = ROOT_DIR / "files/messages.pkl"

template = (
    CHATGPT_ROLE
    + """Answer the question step by step. 
    {conversation_history}
    user: {question}
    assistant:
    """
)
