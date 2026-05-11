import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

USE_REAL_LLM = os.getenv("USE_REAL_LLM", "1") == "1"
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "distilgpt2")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "32"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.2"))
HF_DEVICE = os.getenv("HF_DEVICE")

STUB_VOCAB_SIZE = 4096
STUB_MAX_LENGTH = 64
STUB_HIDDEN_DIM = 128

class TorchStubLLM(torch.nn.Module):
    def __init__(self, vocab_size: int = STUB_VOCAB_SIZE, hidden_dim: int = STUB_HIDDEN_DIM):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim // 2)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim // 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = x.mean(dim=1)
        return self.fc(x)


def _stub_tokenize(prompt: str, max_length: int = STUB_MAX_LENGTH, vocab_size: int = STUB_VOCAB_SIZE):
    ids = [ord(ch) % vocab_size for ch in prompt][:max_length]
    if len(ids) < max_length:
        ids += [0] * (max_length - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def _stub_answer(query: str, logits: torch.Tensor) -> str:
    # Use the stub model output only to exercise PyTorch compute.
    _, token_ids = logits.topk(4, dim=-1)
    tokens = token_ids[0].tolist()
    suffix = " ".join(f"tok{tid}" for tid in tokens)
    return f"PyTorch stub response for '{query}'. Signal: {suffix}"


if USE_REAL_LLM:
    # Load model and tokenizer (done once for efficiency)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use GPU if available
    device = torch.device(HF_DEVICE or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    model.eval()
else:
    tokenizer = None
    model = TorchStubLLM()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device('cpu')
    model.to(device)
    model.eval()


def run_llm(query, context):
    if not USE_REAL_LLM:
        prompt = f"Context: {context}\nQ: {query}\nA:"
        inputs = _stub_tokenize(prompt)
        inputs = inputs.to(device)

        with torch.no_grad():
            logits = model(inputs)
            _ = logits.mean().item()

        return _stub_answer(query, logits)

    # Prepare input prompt
    prompt = f"Context: {context}\nQ: {query}\nA:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    generation_kwargs = {
        "max_new_tokens": HF_MAX_NEW_TOKENS,
        "do_sample": HF_TEMPERATURE > 0,
        "num_return_sequences": 1,
        "no_repeat_ngram_size": 2,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if HF_TEMPERATURE > 0:
        generation_kwargs["temperature"] = HF_TEMPERATURE

    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            **generation_kwargs
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer part
    answer_start = response.rfind("A:")
    answer = response[answer_start + len("A:"):].strip() if answer_start != -1 else response.strip()
    
    return answer if answer else f"Generated response to '{query}' based on context."
