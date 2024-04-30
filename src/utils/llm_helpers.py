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
