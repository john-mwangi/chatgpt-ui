from pathlib import Path

PKG_DIR = Path(__file__).parent.parent.resolve()
costs_path = PKG_DIR / "files/costs.pkl"
msgs_path = PKG_DIR / "files/messages.pkl"

GPT_ROLE = "You are an expert Python programmer."

template = (
    GPT_ROLE
    + """Answer the question step by step. 
    {conversation_history}
    user: {question}
    assistant:
    """
)
