from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Sample documents for RAG
documents = [
    "Large Language Models (LLMs) are powerful AI systems capable of understanding and generating human-like text.",
    "Retrieval-Augmented Generation (RAG) combines retrieval of relevant information with generative models to improve responses.",
    "Distributed computing allows processing large workloads across multiple machines or GPUs.",
    "Load balancing distributes incoming requests evenly across available servers to optimize performance.",
    "GPU workers handle computationally intensive tasks like model inference in parallel.",
    "Vector databases store embeddings for efficient similarity search in RAG systems.",
    "Transformers are neural network architectures that excel at sequence-to-sequence tasks.",
    "PyTorch is a popular deep learning framework for building and training AI models.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Sentence transformers convert text into dense vector representations for semantic search."
]

# Initialize model and index (done once)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
faiss.normalize_L2(embeddings)  # Normalize for cosine
index.add(embeddings)

def retrieve_context(query, top_k=3):
    # Encode query
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve relevant contexts
    contexts = [documents[i] for i in indices[0]]
    return " ".join(contexts)