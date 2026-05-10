#!/usr/bin/env python3
"""
Generate sample PDF documents for RAG testing.
Usage: python create_sample_pdfs.py
"""

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    
    # Define sample content
    SAMPLE_DOCS = {
        "distributed_systems.pdf": {
            "title": "Distributed Systems Fundamentals",
            "content": """
Distributed systems are computing systems whose components are located on different networked computers, 
which communicate and coordinate their actions by passing messages to one another from any system.

Key Concepts:
1. Scalability: The ability to handle increasing amounts of work by adding resources.
2. Reliability: System continues to function even if some components fail.
3. Performance: Response time and throughput meet requirements.
4. Maintainability: System is easy to modify and improve.

Load Balancing in Distributed Systems:
Load balancing is a technique to distribute workload evenly across multiple computing resources. 
It improves resource utilization, maximizes throughput, minimizes response time, and avoids overload 
on individual resources. Common strategies include Round Robin, Least Connections, and Weighted Distribution.

Fault Tolerance:
Fault tolerance is the ability of a system to continue operating properly in the event of the failure 
of some of its components. In distributed systems, this is achieved through redundancy, replication, 
and recovery mechanisms.

Distributed Databases:
Distributed databases span multiple physical locations and are replicated across multiple sites. 
They provide better performance through parallelism, improved reliability through redundancy, and 
increased availability by having data stored at multiple sites.
"""
        },
        "machine_learning_rag.pdf": {
            "title": "Machine Learning and Retrieval-Augmented Generation",
            "content": """
Retrieval-Augmented Generation (RAG) is a hybrid approach that combines large language models with 
information retrieval systems to improve the quality and accuracy of generated text.

How RAG Works:
1. Query Processing: The user query is processed and converted to embeddings.
2. Retrieval: Relevant documents are retrieved from a knowledge base using similarity search.
3. Context Integration: Retrieved documents are formatted as context for the language model.
4. Generation: The LLM generates a response conditioned on the retrieved context.
5. Response: The final response is returned to the user.

Benefits of RAG:
- Reduced Hallucinations: By grounding responses in retrieved documents, the model is less likely to 
  generate false information.
- Domain-Specific Knowledge: RAG systems can leverage specialized documents specific to a domain.
- Up-to-Date Information: Documents can be updated without retraining the model.
- Explainability: The retrieved documents provide evidence for the generated responses.

Vector Databases and Embeddings:
Embeddings convert text into dense vectors that capture semantic meaning. Vector databases like FAISS 
allow efficient similarity search to retrieve the most relevant documents. The embedding model is crucial 
for RAG performance - better embeddings lead to more relevant retrievals.

Large Language Models:
Large language models like GPT, Qwen, and LLaMA are trained on vast amounts of text data and can 
generate coherent, contextually relevant responses. When combined with RAG, they can leverage external 
knowledge to provide more accurate and grounded responses.
"""
        },
        "gpu_computing.pdf": {
            "title": "GPU Computing and Inference Acceleration",
            "content": """
GPU (Graphics Processing Unit) computing has revolutionized machine learning by providing massive parallelism 
for matrix operations commonly used in neural networks.

GPU Architecture:
GPUs contain thousands of small cores optimized for parallel operations. Unlike CPUs which have fewer cores 
optimized for sequential processing, GPUs excel at tasks that can be parallelized. This makes them ideal 
for matrix multiplications, convolutions, and other operations in deep learning.

Benefits of GPU Computing:
1. Speedup: GPU inference is often 10-100x faster than CPU inference for deep learning models.
2. Throughput: GPUs can process multiple requests in parallel.
3. Cost-Effectiveness: GPUs provide better performance per dollar for certain workloads.
4. Energy Efficiency: Modern GPUs like NVIDIA A100 provide excellent performance-per-watt.

GPU Worker Nodes:
In distributed systems, GPU worker nodes handle computationally intensive tasks. Load balancers route 
inference requests to available GPU workers. Multiple GPU nodes can be combined for increased throughput 
and redundancy.

CUDA and PyTorch:
CUDA is NVIDIA's parallel computing platform. PyTorch provides seamless integration with CUDA for GPU-accelerated 
deep learning. Models loaded on GPU devices (cuda:0) can perform inference orders of magnitude faster than on CPU.

Inference Optimization:
Techniques like model quantization, batch processing, and operator fusion can further accelerate GPU inference. 
Batch processing is particularly important - processing multiple requests together is more efficient than 
processing them individually.
"""
        }
    }

    def create_sample_pdfs():
        """Create sample PDF files for testing RAG."""
        from pathlib import Path
        
        output_dir = Path(__file__).parent
        
        for filename, doc_info in SAMPLE_DOCS.items():
            filepath = output_dir / filename
            print(f"Creating {filename}...")
            
            doc = SimpleDocTemplate(str(filepath), pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Add title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                textColor='#1f4788',
                spaceAfter=30,
                alignment=1
            )
            story.append(Paragraph(doc_info['title'], title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Add content
            normal_style = styles['Normal']
            for paragraph_text in doc_info['content'].strip().split('\n\n'):
                if paragraph_text.strip():
                    story.append(Paragraph(paragraph_text.strip(), normal_style))
                    story.append(Spacer(1, 0.15*inch))
            
            doc.build(story)
            print(f"✓ Created {filename}")
        
        print(f"\nAll sample PDFs created in {output_dir}")
        print("The RAG system will automatically load these PDFs when it starts.")

    if __name__ == "__main__":
        try:
            create_sample_pdfs()
        except Exception as e:
            print(f"Error: {e}")
            print("\nTo use this script, install reportlab:")
            print("pip install reportlab")

except ImportError:
    print("reportlab not installed. Sample PDFs will be created manually.")
    print("Install with: pip install reportlab")
