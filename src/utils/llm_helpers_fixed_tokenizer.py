import sys
sys.path.append("..")
from definitions import *


def calculate_token_interest_probs(
    input_prompt: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: tp.Union[transformers.PreTrainedModel, peft.peft_model.PeftModelForCausalLM],
) -> Dict[str, float]:
    assert isinstance(input_prompt, str)
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits  # shape (batch_size, sequence_length, vocab_size)
    next_token_logits = logits[:, -1, :]  # shape (batch_size, vocab_size)

    next_token_logits = next_token_logits.flatten()
    assert next_token_logits.shape == torch.Size((model.config.vocab_size, ))

    next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1).cpu()  # all probs over vocab
    assert torch.isclose(next_token_probs.sum(), torch.tensor(1.0).to(next_token_probs.dtype), atol=1e-03)  # dtype for half/nothalf, -03 for float16
    
    tokens_of_interest = [
        tokenizer("A", add_special_tokens=False).input_ids[-1],
        tokenizer("B", add_special_tokens=False).input_ids[-1],
        tokenizer("C", add_special_tokens=False).input_ids[-1],
        tokenizer("D", add_special_tokens=False).input_ids[-1],
    ]
    
    probs = next_token_probs[tokens_of_interest].tolist()
    res = dict(zip(["A", "B", "C", "D"], probs))
    return res


def get_answer(probs: Dict[str, float]):
    answer, max_prob = 0, 0
    for k, v in probs.items():
        if v > max_prob:
            max_prob = v
            answer = k
    return answer


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2":(
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_input_llama2": (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} \n{input} [/INST]"
    ),
    "prompt_llama2": "[INST]{instruction}[/INST]",
    "prompt_input_diploma_special":(
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nBelow is a diploma text. Your task is to generate abstract of this diploma.\n\n### Input:\n{input}\n\n"
    ),
}


from typing import Dict, Optional, Sequence

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = []
    for text in tqdm(strings, desc="Texts..."):
        tokenized_list.append(tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ))
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


IGNORE_INDEX = -100


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    targets_tokenized = _tokenize_fn(targets, tokenizer)
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    input_ids = [] 
    for example_input_id, target_input_id, example_len, target_len in zip(examples_tokenized["input_ids"], targets_tokenized["input_ids"], examples_tokenized["input_ids_lens"], targets_tokenized["input_ids_lens"]):
        limit = tokenizer.model_max_length
        res = example_input_id
        if example_len == limit:
            res = example_input_id.tolist()[:-target_len] + target_input_id.tolist()[:target_len]
        input_id = torch.tensor(res, dtype=torch.int)
        input_id = input_id.type(torch.LongTensor)
        input_ids.append(input_id)
    labels = copy.deepcopy(input_ids)
    for label, example_len, target_len in zip(labels, examples_tokenized["input_ids_lens"], targets_tokenized["input_ids_lens"]):
        ignore_end = example_len - target_len
        label[:ignore_end] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def get_prefix_len_and_tokens(tokenizer, row, diploma_prefix_len):
    prompt_input_diploma = PROMPT_DICT["prompt_input_diploma_special"]
    source = prompt_input_diploma.format(input=row["diploma"][:diploma_prefix_len])

    target = f"### Response:{row['abstract']}{tokenizer.eos_token}"

    data_dict = preprocess([source], [target], tokenizer)
    
    prefix_len = np.sum(np.array(data_dict["labels"][0]) == IGNORE_INDEX)
    prefix_tokens = data_dict["input_ids"][0][:prefix_len]

    return prefix_len, prefix_tokens


def get_some_model_result(some_model, tokenizer, row, device, diploma_prefix_len):
    prefix_len, prefix_tokens = get_prefix_len_and_tokens(tokenizer, row, diploma_prefix_len)
    some_model.eval()
    generated = some_model.generate(input_ids=prefix_tokens.reshape((1, -1)).to(device), do_sample=False, num_beams=1)
    generated_continue = tokenizer.decode(generated.to('cpu').flatten()[prefix_len:])
    return generated_continue
