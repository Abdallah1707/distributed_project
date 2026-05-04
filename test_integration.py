from rag.retriever import retrieve_context
from llm.inferance import run_llm

# Test RAG
query = "What is RAG?"
context = retrieve_context(query)
print(f"Retrieved context: {context}")

# Test LLM
response = run_llm(query, context)
print(f"LLM response: {response}")