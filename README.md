# **Chatbot**

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that summarizes content and answers questions from specific trained PDFs. It handles both scanned and digital PDFs and can scale to large datasets. The pipeline features chat memory, query decomposition, hybrid search (keyword + semantic), reranking, metadata filtering, and optimized text extraction.

---

## **Table of Contents**
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Models and Libraries](#models-and-libraries)
6. [Evaluation](#evaluation)
7. [Future Improvements](#future-improvements)

---

## **Features**

- **Multilingual Text Extraction**: Supports extraction from both scanned and digitally-created PDFs using OCR and standard parsing techniques.
- **Hybrid Search**: Combines keyword-based and semantic search to improve the relevance of retrieved documents.
- **Optimized Chunking**: Efficiently splits text into chunks to maintain context and ensure accurate embedding.
- **RAG Pipeline**: Retrieves and generates responses based on relevant document chunks.
- **Chat Memory**: Maintains context over multiple queries.
- **Query Decomposition**: Breaks down complex queries into manageable sub-queries.
- **Metadata Filtering**: Filters documents based on language, document type, or metadata attributes.

---

## **Installation**

### **1. Clone the Repository**

```bash
git clone https://github.com/Adii265/ChatBot.git
cd ChatBot
```

2. Install Dependencies:

1. langchain
2. transformers
3. sentence-transformers
4. faiss-cpu
5. tesseract
6. PyMuPDF
7. openai

### Usage:
1. Text Extraction from PDFs
Use the following snippet to extract text from both scanned and digital PDFs

```bash
from pdf_extraction import extract_text

# Extract text from a digitally-created PDF
pdf_path = "sample_digital.pdf"
text = extract_text(pdf_path, ocr=False)

# Extract text from a scanned PDF using OCR
pdf_path_scanned = "sample_scanned.pdf"
text_ocr = extract_text(pdf_path_scanned, ocr=True)

print(text)
print(text_ocr)
```

2. Create Embeddings for Search
   
```bash
from embedding import create_embeddings

# Create embeddings for the extracted text
documents = ["This is a sample document.", "Another document content."]
embeddings = create_embeddings(documents)

print(embeddings)

```

3. RAG Pipeline
   
```bash
from rag_pipeline import RAGPipeline

# Initialize the RAG Pipeline
rag_pipeline = RAGPipeline()

# Ask a question
query = "Summarize the document in English."
response = rag_pipeline.answer_query(query, pdf_path="sample_digital.pdf")

print(response)

```

4. Metadata Filtering
   
```bash
# Add metadata filtering to limit search results by language
response = rag_pipeline.answer_query(
    query="Summarize the document",
    pdf_path="sample_digital.pdf",
    metadata={"language": "English"}
)

print(response)

```

## Pipeline Architecture

### 1. Text Extraction
- **OCR for Scanned PDFs**: Utilizes **Tesseract** for extracting text from scanned documents.
- **PDF Parsing**: Employs **PyMuPDF (Fitz)** for extracting text from digitally created PDFs.

### 2. Embedding Creation
- **Model**: Uses **all-MiniLM-L6-v2** from **sentence-transformers** to generate semantic embeddings.

### 3. Hybrid Search
- **Vector Database**: Implements **FAISS** for efficient semantic search.
- **Keyword Search**: Based on extracted metadata to improve search accuracy.

### 4. Query Decomposition
- Breaks down complex queries into smaller, manageable sub-queries for more granular and relevant results.

### 5. RAG Pipeline
- **LLM**: Integrates **Llama2-7B** for fluent and contextually relevant text generation.



## Models and Libraries

### Text Extraction
- **Tesseract**: Utilized for OCR in scanned PDFs.
- **PyMuPDF (Fitz)**: Used for text extraction from digitally created PDFs.

### Embeddings
- **all-MiniLM-L6-v2**: Sourced from **sentence-transformers** for generating semantic embeddings.

### Language Model
- **Llama2-7B**: Chosen for fluent text generation and summarization tasks.

### Vector Search
- **FAISS**: Implemented for high-performance vector search capabilities.



### Evaluation

**Performance Metrics**
1. Query Relevance: The RAG system returns highly relevant results based on semantic and keyword search.
2. Latency: Achieved an average latency of 2-3 seconds per query.
3. Scalability: The pipeline handles large datasets up to 1TB with stable performance.

**Results**

1. Accurate text extraction from both scanned and digital PDFs.
2. Scalable RAG pipeline with support for multilingual documents.
3. Effective memory retention for chat-like interactions.

Contact
For questions or support, reach out at: aditiagrawal267@gmail.com




