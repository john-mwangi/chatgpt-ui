import yaml
from pydantic_settings import BaseSettings

from chatgpt_ui.configs import PKG_DIR


class Settings(BaseSettings):
    models: list
    pricing: dict
    gpt_params: dict

    def load():
        settings_path = PKG_DIR / "configs/settings.yaml"
        with open(settings_path, mode="r") as f:
            settings = yaml.safe_load(f)

        return Settings(**settings)


if __name__ == "__main__":
    print(Settings.load().models)
    print(Settings.load().pricing)
    print(Settings.load().gpt_params)
