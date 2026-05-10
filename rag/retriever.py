from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from pathlib import Path
import PyPDF2


def load_documents_from_pdfs(pdf_dir: str = None, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Load documents from PDF files in a directory.
    
    Args:
        pdf_dir: Directory containing PDF files. Defaults to 'rag/documents/'
        chunk_size: Approximate size of text chunks in characters
        chunk_overlap: Overlap between chunks for context preservation
    
    Returns:
        List of text chunks from all PDFs
    """
    if pdf_dir is None:
        pdf_dir = os.path.join(os.path.dirname(__file__), 'documents')
    
    documents = []
    pdf_path = Path(pdf_dir)
    
    # Check if directory exists
    if not pdf_path.exists():
        print(f"Warning: PDF directory '{pdf_dir}' not found. Using fallback documents.")
        return get_fallback_documents()
    
    pdf_files = list(pdf_path.glob('*.pdf'))
    
    if not pdf_files:
        print(f"Warning: No PDF files found in '{pdf_dir}'. Using fallback documents.")
        return get_fallback_documents()
    
    print(f"Loading {len(pdf_files)} PDF file(s)...")
    
    for pdf_file in pdf_files:
        try:
            print(f"  Reading: {pdf_file.name}")
            with open(pdf_file, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    try:
                        text += page.extract_text()
                    except Exception as e:
                        print(f"    Warning: Failed to extract page {page_num} from {pdf_file.name}: {e}")
            
            # Split text into chunks with overlap
            if text.strip():
                chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
                documents.extend(chunks)
                print(f"    Extracted {len(chunks)} chunks")
            else:
                print(f"    Warning: No text extracted from {pdf_file.name}")
        
        except Exception as e:
            print(f"  Error processing {pdf_file.name}: {e}")
    
    if not documents:
        print("No documents extracted from PDFs. Using fallback documents.")
        return get_fallback_documents()
    
    print(f"Total chunks loaded: {len(documents)}\n")
    return documents


def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Split text into overlapping chunks to preserve context.
    
    Args:
        text: Text to split
        chunk_size: Approximate size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = text.rfind('.', start, end)
            if last_period > start:
                end = last_period + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - chunk_overlap
    
    return chunks


def get_fallback_documents() -> list:
    """
    Fallback sample documents when no PDFs are available.
    """
    return [
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


# Load documents from PDFs on startup
documents = load_documents_from_pdfs()

# Initialize model and index (done once)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
faiss.normalize_L2(embeddings)  # Normalize for cosine
index.add(embeddings)

print(f"RAG system initialized with {len(documents)} documents/chunks\n")


def retrieve_context(query, top_k=3):
    """
    Retrieve relevant context for a query using semantic similarity.
    
    Args:
        query: User query string
        top_k: Number of top results to retrieve
    
    Returns:
        Concatenated string of relevant document chunks
    """
    # Encode query
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve relevant contexts
    contexts = [documents[i] for i in indices[0]]
    return " ".join(contexts)