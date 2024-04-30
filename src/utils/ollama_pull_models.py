import sys
sys.path.append("..")
from definitions import *


def pull_model(model_name: str):
    subprocess.run(["ollama", "pull", model_name])


def pull_models(model_names: List[str]):
    for model_name in model_names:
        pull_model(model_name)


def main():
    model_names = OLLAMA_MODEL_TO_NAME.values()
    pull_models(model_names)


if __name__ == "__main__":
    main()
