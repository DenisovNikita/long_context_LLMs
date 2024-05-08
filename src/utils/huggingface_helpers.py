import sys
sys.path.append("..")
from definitions import *


def get_tokenizer(model: str, repo: str):
    return AutoTokenizer.from_pretrained(repo, cache_dir="/home/jupyter/work/resources/modelcache/")


def get_model(model: str, repo: str):
    return AutoModelForCausalLM.from_pretrained(repo, cache_dir="/home/jupyter/work/resources/modelcache/")
    # config = AutoConfig.from_pretrained(repo)
    # return AutoModel.from_config(config)


def get_num_tokens(tokenizer, text: str):
    return len(tokenizer(text)["input_ids"])


def process_texts_with_tokenizers(input_dir_path: Path, output_dir_path: Path, model_to_repo: Dict[str, str], debug=False):
    output_dir_path.mkdir(exist_ok=True, parents=True)
    failed = []
    for model, repo in tqdm(model_to_repo.items(), desc="Iterating through models"):
        try:
            tokenizer = get_tokenizer(model, repo) 
            book_to_tokens = dict()
            for file in input_dir_path.glob("*.txt"):
                with open(file, "r") as f:
                    book_to_tokens[file.stem] = get_num_tokens(tokenizer, f.read())
            with open(output_dir_path.joinpath(f"{model}.json"), "w") as f:
                json.dump(book_to_tokens, f, indent=2, ensure_ascii=False)
        except Exception as e:
            if debug:
                print(f"failed {model} with exception {e}")
            failed.append(model)
    if failed:
        print(f"Failed: {failed}.\nCheck that you logged in to hugginface and have permissions for those models.")
    else:
        print("All ok")
