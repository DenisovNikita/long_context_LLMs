import sys
sys.path.append("..")
from definitions import *
sys.path.append(DIPLOMA_DIR_PATH.joinpath("llama3/llama/").as_posix())
from tokenizer import Tokenizer


def process_texts_with_llama3_tokenizer(input_dir_path: Path, output_dir_path: Path):
    tokenizer = Tokenizer(DIPLOMA_DIR_PATH.joinpath("llama3/Meta-Llama-3-8B/tokenizer.model").as_posix())
    book_to_tokens = dict()
    for file in input_dir_path.glob("*.txt"):
        with open(file, "r") as f:
            num_tokens = len(tokenizer.encode(f.read(), bos=False, eos=False))
        book_to_tokens[file.stem] = num_tokens
    with open(output_dir_path.joinpath("llama-3-8b.json"), "w") as f:
        json.dump(book_to_tokens, f, indent=2, ensure_ascii=False)