import os
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

USE_REAL_LLM = os.getenv("USE_REAL_LLM", "1") == "1"

if USE_REAL_LLM:
    # Load model and tokenizer (done once for efficiency)
    model_name = 'distilgpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
else:
    tokenizer = None
    model = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_llm(query, context):
    if not USE_REAL_LLM:
        # Fast simulation mode lets the distributed system be tested at 1000+
        # requests without waiting for hundreds of local GPT-2 generations.
        time.sleep(0.02)
        return f"Simulated answer for '{query}' using retrieved context: {context[:160]}"

    # Prepare input prompt
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=30,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer part
    answer_start = response.find("Answer:") + len("Answer:")
    answer = response[answer_start:].strip()
    
    return answer if answer else f"Generated response to '{query}' based on context."
