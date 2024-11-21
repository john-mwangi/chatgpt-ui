from pathlib import Path

PKG_DIR = Path(__file__).parent.parent.resolve()
costs_path = PKG_DIR / "files/costs.pkl"
msgs_path = PKG_DIR / "files/messages.pkl"

GPT_ROLE = "You are a helpful assistant."

template = (
    GPT_ROLE
    + """Answer the question step by step. 
    human: {question}
    assistant:
    """
)
