# 📚 Personal RAG Assistant with Local LLM

A **Retrieval-Augmented Generation (RAG)** application that allows users to upload and chat with their own **PDF, TXT, and Markdown documents** using a locally running Large Language Model.

The application retrieves relevant information from uploaded documents using **semantic search** and generates context-aware answers using **Llama 3.1 8B** through **Ollama**.

A **Streamlit web interface** provides document uploading, processing, conversational chat, and source-page references.

The entire LLM pipeline can run locally, so no paid LLM API is required.

---

## ✨ Features

- 📄 Upload PDF, TXT, and Markdown documents
- ✂️ Split large documents into smaller overlapping chunks
- 🔎 Generate semantic embeddings using `all-MiniLM-L6-v2`
- 💾 Store document embeddings in ChromaDB
- 🧠 Perform semantic similarity search over documents
- 🤖 Generate answers using Llama 3.1 8B through Ollama
- 💬 Maintain conversation memory for follow-up questions
- 📚 Display source documents and PDF page numbers
- 🌐 Streamlit-based interactive web interface
- 🗣️ Respond in the same language as the user's question
- 🔐 Local LLM inference for improved document privacy
- 💰 No paid LLM API required

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| LangChain | RAG pipeline and conversational retrieval |
| Streamlit | Interactive web interface |
| ChromaDB | Vector database |
| HuggingFace Sentence Transformers | Document embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| Ollama | Local LLM runtime |
| Llama 3.1 8B | Local language model |
| PyPDF | PDF document loading and extraction |

---

## 🏗️ Architecture

```text
                  User Documents
                PDF / TXT / Markdown
                       │
                       ▼
                Document Loader
                       │
                       ▼
          RecursiveCharacterTextSplitter
                       │
                       ▼
               Document Chunks
                       │
                       ▼
              all-MiniLM-L6-v2
             HuggingFace Embeddings
                       │
                       ▼
                  ChromaDB
                Vector Store
                       │
                       ▼
                 User Question
                       │
                       ▼
               Semantic Search
                       │
                       ▼
          Top Relevant Document Chunks
                       │
                       ▼
               Retrieved Context
                       │
                       ▼
             LangChain RAG Pipeline
                       │
                       ▼
              Llama 3.1 8B
                  via Ollama
                       │
                       ▼
                Generated Answer
                       │
                       ▼
           Answer + Document Sources
                       │
                       ▼
                Streamlit UI
```

---

## 🔍 How It Works

### 1. Document Loading

Users can upload:

- PDF
- TXT
- Markdown

documents through the Streamlit interface.

The documents are stored in the local `docs/` directory.

`PyPDFLoader` is used for PDF files and `TextLoader` is used for text-based files.

### 2. Text Chunking

Large documents cannot efficiently be sent directly to an LLM.

The application uses:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

Each document is divided into chunks of approximately **1000 characters**, with a **200-character overlap** between consecutive chunks.

The overlap helps preserve contextual information across chunk boundaries.

### 3. Embedding Generation

Each chunk is converted into a numerical vector using:

```text
all-MiniLM-L6-v2
```

The embedding model converts semantically similar text into vectors that are located close to each other in vector space.

### 4. Vector Storage

The generated embeddings are stored in **ChromaDB**.

ChromaDB acts as the application's vector database and enables efficient similarity-based retrieval.

### 5. Semantic Retrieval

When the user asks a question, the application searches ChromaDB for the most relevant document chunks.

The retriever currently uses:

```python
search_kwargs={"k": 4}
```

Therefore, the top **4 relevant chunks** are supplied as context to the language model.

### 6. Local LLM Generation

The retrieved context and user question are passed to:

```text
Llama 3.1 8B
```

The model runs locally through **Ollama**.

The application uses a custom prompt instructing the model to answer based on the retrieved document context and respond in the same language as the user's question.

### 7. Conversation Memory

The application uses:

```text
ConversationBufferWindowMemory
```

with a window of recent conversation messages.

This allows users to ask follow-up questions while maintaining conversational context.

### 8. Source References

The application returns the retrieved source documents along with the generated answer.

For PDF documents, the UI can display information such as:

```text
javabook.pdf — Page 120
```

This makes it easier to understand where the retrieved information originated.

---

# 🚀 Running the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/yanshuman/personal-rag-assistant.git
cd personal-rag-assistant
```

---

## 2. Create a Virtual Environment

The project was developed using **Python 3.12**.

On Windows:

```powershell
py -3.12 -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Verify:

```powershell
python --version
```

Example:

```text
Python 3.12.x
```

---

## 3. Install Dependencies

Install the required Python packages:

```powershell
pip install langchain langchain-community langchain-text-splitters langchain-ollama chromadb sentence-transformers pypdf streamlit
```

Verify that there are no dependency conflicts:

```powershell
pip check
```

---

## 4. Install Ollama

Install Ollama on your system and verify the installation:

```powershell
ollama --version
```

Then download the Llama model:

```powershell
ollama pull llama3.1:8b
```

Check installed models:

```powershell
ollama list
```

You should see:

```text
NAME
llama3.1:8b
```

---

## 5. Add Documents

You have two options.

### Option A — Streamlit UI

Start the Streamlit application and upload documents directly through the sidebar.

### Option B — Manually

Place PDF, TXT, or Markdown documents inside:

```text
docs/
```

Example:

```text
personal-rag-assistant/
│
├── app.py
├── rag.py
├── README.md
├── .gitignore
│
└── docs/
    └── example.pdf
```

---

# 🖥️ Run the Streamlit Application

Start the web application:

```powershell
streamlit run app.py
```

Streamlit will display a local address similar to:

```text
http://localhost:8501
```

Open it in your browser.

Then:

1. Open the sidebar.
2. Upload your documents.
3. Click **Process Documents**.
4. Wait for the vector database to be created.
5. Ask questions about your documents.

---

# 💻 Run the Command-Line Version

The original command-line RAG assistant can also be started using:

```powershell
python -u rag.py
```

Example:

```text
RAG Assistant ready. Type your questions (type 'quit' to exit):

You: What is inheritance in Java?

Assistant: Inheritance in Java is an object-oriented programming
mechanism that allows one class to acquire the properties and
behaviors of another class.
```

To exit:

```text
quit
```

---

# 💬 Example Usage

### User

```text
What is inheritance in Java?
```

### RAG Pipeline

```text
Question
   ↓
Embedding
   ↓
ChromaDB similarity search
   ↓
Top 4 relevant chunks
   ↓
Context + Question
   ↓
Llama 3.1 8B
```

### Assistant

```text
Inheritance in Java is an object-oriented programming mechanism
where one class can acquire the properties and methods of another
class.
```

The Streamlit interface also displays the document sources used to generate the response.

---

# 📂 Project Structure

```text
personal-rag-assistant/
│
├── app.py
│   └── Streamlit web interface and RAG pipeline
│
├── rag.py
│   └── Command-line RAG assistant
│
├── README.md
│   └── Project documentation
│
├── .gitignore
│   └── Files excluded from Git
│
├── docs/
│   └── User documents
│
├── chroma_db/
│   └── Local ChromaDB vector database
│
└── venv/
    └── Python virtual environment
```

The `docs/`, `chroma_db/`, and `venv/` directories should normally be excluded from Git.

---

# 🔐 Privacy

The project is designed to support local document processing and local LLM inference.

The Llama model runs locally through Ollama, so the application does not require sending prompts to a paid cloud LLM API.

Sensitive local project data should be excluded from Git using `.gitignore`, including:

```text
venv/
chroma_db/
docs/
.env
__pycache__/
*.pyc
```

Users should still review their environment and dependencies when working with sensitive documents.

---

# ⚙️ Key Configuration

### Chunk Size

```python
chunk_size=1000
```

### Chunk Overlap

```python
chunk_overlap=200
```

### Number of Retrieved Chunks

```python
k=4
```

### LLM

```python
ChatOllama(
    model="llama3.1:8b",
    temperature=0.2
)
```

### Embedding Model

```text
all-MiniLM-L6-v2
```

### Conversation Memory

```python
ConversationBufferWindowMemory(
    k=10
)
```

These parameters can be adjusted depending on document size, hardware resources, retrieval quality, and desired response behavior.

---

# 🎯 Key Learning Outcomes

This project demonstrates practical experience with:

- Retrieval-Augmented Generation (RAG)
- Large Language Models
- Local LLM deployment
- Vector databases
- Semantic search
- Text embeddings
- Document chunking
- Prompt engineering
- Conversational memory
- LangChain
- Streamlit application development
- Ollama
- Source-aware document question answering

---

# 🔮 Future Improvements

Potential improvements include:

- Incremental document indexing instead of rebuilding existing embeddings
- Document deletion from the UI
- Multiple knowledge bases
- Improved metadata filtering
- Hybrid keyword + semantic search
- Reranking retrieved documents
- Streaming LLM responses
- Configurable chunk size and retrieval count
- Support for DOCX and additional document formats
- Better citation formatting
- Docker support
- Automated evaluation of retrieval quality
- Migration from deprecated LangChain components to newer APIs

---

# ⚠️ Limitations

- Response quality depends on the quality of retrieved document chunks.
- Large documents may require significant initial embedding time.
- Llama 3.1 8B performance depends on the available system hardware.
- The current implementation can re-index documents when processing is triggered again.
- Generated answers should be verified against the displayed source documents for important use cases.

---

# 🙏 Acknowledgment

This project was adapted and extended from the **Personal RAG Assistant** project in the `blurred-machine/ai-weekend-builds` repository.

The implementation was extended to:

- Use a locally running **Llama 3.1 8B** model through Ollama
- Remove the requirement for a paid Anthropic API
- Add a **Streamlit web interface**
- Support document uploading through the browser
- Display retrieved document sources and PDF page numbers
- Add custom response-language instructions
- Improve local document-based conversational interaction

---

## ⭐ About the Project

This project was built as a hands-on implementation of a modern **Retrieval-Augmented Generation pipeline**, combining semantic retrieval with a locally hosted LLM to create a private document-question-answering assistant.
