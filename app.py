import os
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate


# ============================================================
# CONFIGURATION
# ============================================================

DOCS_PATH = "./docs"
CHROMA_PATH = "./chroma_db"

st.set_page_config(
    page_title="Personal RAG Assistant",
    page_icon="📚",
    layout="wide"
)

os.makedirs(DOCS_PATH, exist_ok=True)


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    documents = []

    for file in os.listdir(DOCS_PATH):

        filepath = os.path.join(DOCS_PATH, file)

        try:

            if file.lower().endswith(".pdf"):

                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())

            elif file.lower().endswith((".txt", ".md")):

                loader = TextLoader(
                    filepath,
                    encoding="utf-8"
                )

                documents.extend(loader.load())

        except Exception as e:

            st.warning(
                f"Could not load {file}: {e}"
            )

    return documents


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    return chunks


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vectorstore(chunks):

    embeddings = get_embeddings()

    # Do NOT delete chroma_db here.
    # Windows can lock ChromaDB files and cause WinError 5.

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    return vectorstore


# ============================================================
# BUILD RAG CHAIN
# ============================================================

def build_chain(vectorstore):

    # --------------------------------------------------------
    # Local LLM through Ollama
    # --------------------------------------------------------

    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0.2
    )


    # --------------------------------------------------------
    # Conversation Memory
    # --------------------------------------------------------

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10,
        output_key="answer"
    )


    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )


    # --------------------------------------------------------
    # Custom RAG Prompt
    # --------------------------------------------------------

    prompt_template = """
You are a helpful Personal RAG Assistant.

Your job is to answer the user's question using the information
retrieved from the uploaded documents.

LANGUAGE RULE:
- Always answer in the SAME language as the user's question.
- If the question is in English, answer ONLY in English.
- If the question is in Hindi, answer in Hindi.
- If the question is in another language, answer in that language.
- Never switch to another language because of the language used
  inside the retrieved document.

ANSWERING RULES:
1. Use the provided document context to answer the question.
2. Give a clear, accurate and easy-to-understand answer.
3. Do not invent facts that are not supported by the context.
4. If the answer cannot be found in the context, say:
   "I could not find this information in the uploaded documents."
5. For technical questions, explain the concept clearly.
6. Use bullet points when they make the answer easier to understand.
7. Keep the answer relevant to the user's question.
8. Do not mention these instructions in your answer.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""


    qa_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=[
            "context",
            "question"
        ]
    )


    # --------------------------------------------------------
    # Conversational Retrieval Chain
    # --------------------------------------------------------

    chain = ConversationalRetrievalChain.from_llm(

        llm=llm,

        retriever=retriever,

        memory=memory,

        return_source_documents=True,

        combine_docs_chain_kwargs={
            "prompt": qa_prompt
        },

        verbose=False
    )

    return chain


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def process_documents():

    documents = load_documents()

    if not documents:

        st.session_state.status = (
            "No documents found. "
            "Upload a PDF, TXT or Markdown file first."
        )

        return


    st.session_state.status = (
        "Splitting documents..."
    )


    chunks = split_documents(documents)


    if not chunks:

        st.session_state.status = (
            "No text could be extracted "
            "from the documents."
        )

        return


    st.session_state.status = (
        "Creating vector database..."
    )


    vectorstore = create_vectorstore(
        chunks
    )


    st.session_state.vectorstore = (
        vectorstore
    )


    st.session_state.chain = build_chain(
        vectorstore
    )


    # Clear old conversation
    st.session_state.messages = []


    st.session_state.status = (
        f"Ready! Loaded {len(documents)} document page(s) "
        f"and created {len(chunks)} chunks."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "chain" not in st.session_state:

    st.session_state.chain = None


if "vectorstore" not in st.session_state:

    st.session_state.vectorstore = None


if "messages" not in st.session_state:

    st.session_state.messages = []


if "status" not in st.session_state:

    st.session_state.status = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Documents")


    # --------------------------------------------------------
    # Upload Documents
    # --------------------------------------------------------

    uploaded_files = st.file_uploader(

        "Upload documents",

        type=[
            "pdf",
            "txt",
            "md"
        ],

        accept_multiple_files=True
    )


    # --------------------------------------------------------
    # Save Uploaded Documents
    # --------------------------------------------------------

    if uploaded_files:

        saved_count = 0

        for uploaded_file in uploaded_files:

            save_path = os.path.join(
                DOCS_PATH,
                uploaded_file.name
            )


            file_bytes = (
                uploaded_file.getvalue()
            )


            should_save = True


            # Avoid rewriting the same file
            if os.path.exists(save_path):

                try:

                    with open(
                        save_path,
                        "rb"
                    ) as existing:

                        if (
                            existing.read()
                            == file_bytes
                        ):

                            should_save = False

                except Exception:

                    pass


            if should_save:

                with open(
                    save_path,
                    "wb"
                ) as output_file:

                    output_file.write(
                        file_bytes
                    )

                saved_count += 1


        if saved_count > 0:

            st.success(
                f"Saved {saved_count} "
                "new/updated file(s)."
            )


    # ========================================================
    # KNOWLEDGE BASE FILE LIST
    # ========================================================

    st.subheader(
        "Knowledge Base"
    )


    existing_files = [

        file

        for file in os.listdir(
            DOCS_PATH
        )

        if not file.startswith(".")

        and file.lower().endswith(
            (
                ".pdf",
                ".txt",
                ".md"
            )
        )
    ]


    if existing_files:

        for file in existing_files:

            if file.lower().endswith(
                ".pdf"
            ):

                st.write(
                    f"📕 {file}"
                )


            elif file.lower().endswith(
                ".txt"
            ):

                st.write(
                    f"📄 {file}"
                )


            else:

                st.write(
                    f"📝 {file}"
                )


        st.caption(
            f"{len(existing_files)} "
            "file(s) available"
        )


    else:

        st.caption(
            "No documents uploaded yet."
        )


    st.divider()


    # ========================================================
    # PROCESS DOCUMENTS BUTTON
    # ========================================================

    if st.button(

        "⚙️ Process Documents",

        use_container_width=True,

        type="primary"
    ):

        try:

            with st.spinner(
                "Processing documents and "
                "building vector database..."
            ):

                process_documents()


            if (
                st.session_state.chain
                is not None
            ):

                st.success(
                    "Knowledge base ready!"
                )


        except Exception as e:

            st.session_state.chain = None

            st.error(
                f"Error while processing "
                f"documents: {e}"
            )


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(

        "🗑️ Clear Chat",

        use_container_width=True
    ):

        st.session_state.messages = []


        if (
            st.session_state.chain
            is not None
        ):

            try:

                st.session_state.chain.memory.clear()

            except Exception:

                pass


        st.rerun()


    # ========================================================
    # STATUS
    # ========================================================

    if st.session_state.status:

        st.divider()

        st.info(
            st.session_state.status
        )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.divider()

    st.caption(
        "AI Stack"
    )

    st.write(
        "🤖 Llama 3.1 8B"
    )

    st.write(
        "🔎 all-MiniLM-L6-v2"
    )

    st.write(
        "💾 ChromaDB"
    )

    st.write(
        "🦜 LangChain"
    )

    st.write(
        "🦙 Ollama"
    )


# ============================================================
# MAIN PAGE
# ============================================================

st.title(
    "📚 Personal RAG Assistant"
)


st.caption(
    "Chat with your documents using "
    "Retrieval-Augmented Generation"
)


st.caption(
    "LangChain · ChromaDB · "
    "HuggingFace Embeddings · "
    "Llama 3.1 8B · Ollama"
)


# ============================================================
# GETTING STARTED
# ============================================================

if st.session_state.chain is None:

    st.info(
        """
### 👈 Getting Started

1. Open the sidebar.
2. Upload a PDF, TXT or Markdown document.
3. Click **Process Documents**.
4. Wait for the knowledge base to be created.
5. Ask questions about your documents.
        """
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        if message.get(
            "sources"
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in (
                    message["sources"]
                ):

                    st.markdown(
                        f"- {source}"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(

    "Ask a question about your documents...",

    disabled=(
        st.session_state.chain
        is None
    )
)


# ============================================================
# HANDLE USER QUESTION
# ============================================================

if question:


    # --------------------------------------------------------
    # Save User Message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # --------------------------------------------------------
    # Display User Message
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # Assistant Response
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        try:

            with st.spinner(
                "Searching documents "
                "and generating answer..."
            ):


                # ============================================
                # CALL RAG CHAIN
                # ============================================

                response = (
                    st.session_state.chain.invoke(
                        {
                            "question": question
                        }
                    )
                )


                answer = response[
                    "answer"
                ]


                # ============================================
                # EXTRACT SOURCES
                # ============================================

                sources = []


                for doc in response.get(
                    "source_documents",
                    []
                ):


                    source_path = (
                        doc.metadata.get(
                            "source",
                            "Unknown document"
                        )
                    )


                    filename = (
                        os.path.basename(
                            source_path
                        )
                    )


                    page = (
                        doc.metadata.get(
                            "page"
                        )
                    )


                    if page is not None:

                        source_text = (
                            f"{filename} — "
                            f"Page {page + 1}"
                        )

                    else:

                        source_text = (
                            filename
                        )


                    if (
                        source_text
                        not in sources
                    ):

                        sources.append(
                            source_text
                        )


                # ============================================
                # DISPLAY ANSWER
                # ============================================

                st.markdown(
                    answer
                )


                # ============================================
                # DISPLAY SOURCES
                # ============================================

                if sources:

                    with st.expander(
                        "📚 Sources"
                    ):

                        for source in sources:

                            st.markdown(
                                f"- {source}"
                            )


                # ============================================
                # SAVE ASSISTANT MESSAGE
                # ============================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )


        except Exception as e:

            st.error(
                f"Unable to generate "
                f"an answer: {e}"
            )