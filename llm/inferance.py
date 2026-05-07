import os
import threading
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

USE_REAL_LLM = os.getenv("USE_REAL_LLM", "1") == "1"
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "distilgpt2")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "32"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.2"))
HF_DEVICE = os.getenv("HF_DEVICE")
generation_lock = threading.RLock()

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
    model = None
    device = torch.device(HF_DEVICE or ('cuda' if torch.cuda.is_available() else 'cpu'))

def run_llm(query, context):
    if not USE_REAL_LLM:
        # Fast simulation mode lets the distributed system be tested at 1000+
        # requests without waiting for hundreds of local GPT-2 generations.
        time.sleep(0.02)
        return f"Simulated answer for '{query}' using retrieved context: {context[:160]}"

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

    with generation_lock, torch.no_grad():
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
