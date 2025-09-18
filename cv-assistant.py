"""
CV Assistant – Your AI-Powered Resume Helper
===========================================

This script turns your CV and documents into an **interactive AI assistant**.  
Just drop your files (PDF, Word, or text) into the `data/` folder, and the script 
will process them so you can ask natural questions like *“What projects has Saud 
worked on with AI?”* or *“Summarize my data science experience.”*

What it does:
- Reads your documents and splits them into smaller, searchable chunks.
- Detects new or updated files and refreshes the knowledge base automatically.
- Stores everything in a **FAISS index** for fast retrieval.
- Uses **OpenAI GPT** to answer your questions with context from your CV.
- Lets you chat directly in the terminal – type your query, get an answer, 
  and type `exit` when you’re done.

How to use:
1. Create a folder named `data/` in the same directory as this script.
2. Add your CV and related documents (PDF, Word, or text files) to the
3. Make sure your `.env` file has your `OPENAI_API_KEY`.
4. Run the script with:
   ```bash
   python cv-assistant.py
"""

"""
CV Assistant with RAG + Knowledge Graph
---------------------------------------
This script builds an AI-powered CV assistant that can answer recruiter questions.

Features:
1. Uses FAISS vector store + OpenAI embeddings for semantic search (RAG).
2. Builds a lightweight Knowledge Graph (KG) from your documents for structured Q&A.
3. Automatically detects new/updated files via hashing and updates embeddings + KG.
4. Provides an interactive chatbot interface for recruiters.

Author: Saud Ahmad
"""



import os
import json
import pickle
import hashlib
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.chains import RetrievalQA
from langchain.chains import GraphQAChain
from langchain_community.graphs import NetworkxEntityGraph


# =========================
# Helper Functions
# =========================
def file_hash(file_path: Path) -> str:
    """Return SHA256 hash of file contents."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_document(file_path: Path):
    """Load a document based on file extension."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    elif ext == ".txt":
        return TextLoader(str(file_path), encoding="utf-8").load()
    elif ext in [".doc", ".docx"]:
        return UnstructuredWordDocumentLoader(str(file_path)).load()
    else:
        print(f"⚠️ Skipping unsupported file: {file_path}")
        return []


def load_env_variables():
    print("⚙️ Loading environment variables...")
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")


def load_hash_store(path: Path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_hash_store(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_or_create_vectorstore(faiss_path: Path, embeddings, new_docs=None):
    if faiss_path.exists():
        print("📂 Found existing FAISS index, loading it...")
        return FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
    elif new_docs:
        print("🆕 Creating new FAISS index...")
        vectorstore = FAISS.from_documents(new_docs, embeddings)
        vectorstore.save_local(str(faiss_path))
        return vectorstore
    else:
        return None


def process_files(data_dir: Path, embedded_files: dict):
    """Process documents and return new chunks to embed."""
    new_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    for file in data_dir.glob("*"):
        if file.suffix.lower() not in [".pdf", ".txt", ".doc", ".docx"]:
            continue

        h = file_hash(file)

        if file.name not in embedded_files:
            print(f"➕ New file detected: {file.name}")
        elif embedded_files[file.name] != h:
            print(f"🔄 File updated: {file.name}, re-embedding...")
        else:
            print(f"⏩ Skipping unchanged file: {file.name}")
            continue

        docs = load_document(file)
        chunks = splitter.split_documents(docs)
        new_docs.extend(chunks)
        embedded_files[file.name] = h

    return new_docs


def build_domain_kg(docs, kg_path: Path):
    """Build or update the domain knowledge graph."""
    print("🧩 Building Knowledge Graph (domain)...")
    graph = nx.DiGraph()

    for doc in docs:
        text = doc.page_content
        words = text.split()
        if len(words) >= 3:
            graph.add_edge(words[0], words[1], relation="related_to")

    with open(kg_path, "wb") as f:
        pickle.dump(graph, f)

    print("✅ Domain Knowledge Graph built/updated.")
    return NetworkxEntityGraph(graph=graph)


def load_or_init_conversation_memory(path: Path):
    if path.exists():
        with open(path, "rb") as f:
            graph = pickle.load(f)
        print("💬 Loaded existing Conversation Memory KG.")
    else:
        graph = nx.DiGraph()
        print("🆕 Created new Conversation Memory KG.")
    return graph


def init_qa_chains(vectorstore, kg):
    """Initialize RAG and KG QA chains."""
    print("🤖 Initializing retrievers and LLM...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    kg_chain = GraphQAChain.from_llm(llm=llm, graph=kg, verbose=False)

    print("✅ QA chains (RAG + KG) are ready.")
    return qa_chain, kg_chain


def update_conversation_memory(query, result, conversation_graph, conv_memory_path: Path):
    q_node = f"Q: {query}"
    a_node = f"A: {result}"

    conversation_graph.add_node(q_node, type="question")
    conversation_graph.add_node(a_node, type="answer")
    conversation_graph.add_edge(q_node, a_node, relation="answered_by")

    with open(conv_memory_path, "wb") as f:
        pickle.dump(conversation_graph, f)

    print("📝 Conversation Memory KG updated.")


def run_assistant(qa_chain, kg_chain, conversation_graph, conv_memory_path: Path):
    print("\n🚀 CV Assistant Ready! Ask me questions (type 'exit' to quit)")
    print("   (Uses RAG by default. Prefix your query with 'kg:' for Knowledge Graph mode.)")

    while True:
        query = input("\nQuery: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting assistant. Goodbye!")
            break

        print("💭 Thinking...")
        if query.lower().startswith("kg:"):
            result = kg_chain.run(query[3:].strip())
        else:
            result = qa_chain.run(query)

        print("\nAnswer:", result)
        update_conversation_memory(query, result, conversation_graph, conv_memory_path)


# =========================
# Main
# =========================
def main():
    OPENAI_API_KEY = load_env_variables()

    data_dir = Path("data")
    faiss_path = data_dir / "faiss_index"
    hash_store_path = data_dir / "embedded_files.json"
    kg_path = data_dir / "knowledge_graph.pkl"
    conv_memory_path = data_dir / "conversation_memory.pkl"

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-large")
    embedded_files = load_hash_store(hash_store_path)

    # Process files and update vectorstore
    new_docs = process_files(data_dir, embedded_files)
    vectorstore = load_or_create_vectorstore(faiss_path, embeddings, new_docs)

    if new_docs and vectorstore:
        vectorstore.add_documents(new_docs)
        vectorstore.save_local(str(faiss_path))
        print("✅ FAISS index updated with new/changed files.")
    else:
        print("✅ No new or updated files. FAISS index unchanged.")

    save_hash_store(hash_store_path, embedded_files)

    # Build/load Knowledge Graph
    kg = build_domain_kg(new_docs, kg_path) if new_docs else build_domain_kg([], kg_path)

    # Load conversation memory
    conversation_graph = load_or_init_conversation_memory(conv_memory_path)

    # Initialize QA chains
    qa_chain, kg_chain = init_qa_chains(vectorstore, kg)

    # Run assistant loop
    run_assistant(qa_chain, kg_chain, conversation_graph, conv_memory_path)


if __name__ == "__main__":
    main()
