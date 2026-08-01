# Personal RAG Assistant with Local LLM

A Retrieval-Augmented Generation (RAG) application that allows users to
chat with their own PDF and text documents.

The application retrieves relevant information from uploaded documents
using semantic search and generates answers using a locally running
Llama 3.1 8B model through Ollama.

## Features

- Load PDF, TXT, and Markdown documents
- Split documents into smaller chunks
- Generate embeddings using all-MiniLM-L6-v2
- Store and retrieve embeddings using ChromaDB
- Semantic document retrieval
- Local LLM inference using Llama 3.1 8B and Ollama
- Conversation memory for follow-up questions
- No paid LLM API required

## Tech Stack

- Python
- LangChain
- ChromaDB
- HuggingFace Sentence Transformers
- Ollama
- Llama 3.1 8B
- PyPDF

## How It Works

Documents
   ↓
Document Loading
   ↓
Text Chunking
   ↓
HuggingFace Embeddings
   ↓
ChromaDB Vector Store
   ↓
Similarity Search
   ↓
Relevant Context
   ↓
Llama 3.1 8B (Ollama)
   ↓
Generated Answer

## Run Locally

Create and activate a virtual environment:

    py -3.12 -m venv venv
    .\venv\Scripts\Activate.ps1

Install the required dependencies.

Install Ollama and download the model:

    ollama pull llama3.1:8b

Add PDF, TXT, or Markdown files to the `docs/` directory.

Run:

    python rag.py

Then ask questions about your documents.

## Example

    You: What is inheritance in Java?

    Assistant: Inheritance is an object-oriented programming mechanism...

## Privacy

Documents, the vector database, virtual environment, and environment
variables are excluded from Git using `.gitignore`.

## Acknowledgment

This project was adapted and extended from the Personal RAG Assistant
project in the `blurred-machine/ai-weekend-builds` repository.
The implementation was modified to use a locally running Llama 3.1 8B
model through Ollama instead of requiring a paid Anthropic API.
