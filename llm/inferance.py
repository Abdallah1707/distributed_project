from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer (done once for efficiency)
model_name = 'distilgpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def run_llm(query, context):
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
            max_length=inputs['input_ids'].shape[1] + 100,  # Generate up to 100 new tokens
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer part
    answer_start = response.find("Answer:") + len("Answer:")
    answer = response[answer_start:].strip()
    
    return answer if answer else f"Generated response to '{query}' based on context."