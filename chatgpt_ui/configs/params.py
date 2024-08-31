from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

PKG_DIR = Path(__file__).parent.parent.resolve()

CHATGPT_ROLE = "You are an expert Python programmer."

template = (
    CHATGPT_ROLE
    + """Answer the question step by step. 
    {conversation_history}
    user: {question}
    assistant:
    """
)


class Settings(BaseSettings):
    models: list
    pricing: dict

    def load():
        settings_path = PKG_DIR / "configs/settings.yaml"
        with open(settings_path, mode="r") as f:
            settings = yaml.safe_load(f)

        return Settings(**settings)


if __name__ == "__main__":
    print(Settings.load().models)
    print(Settings.load().pricing)
