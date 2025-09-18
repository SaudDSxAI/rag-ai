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
- Stores everything in a **FAISS index** for fast retrieval (RAG).
- Builds a **Knowledge Graph (KG)** from your CV for structured relationships.
- Maintains a **Conversation Memory KG** that stores past Q&A for context-aware follow-ups.
- Uses **OpenAI GPT** to answer your questions with combined knowledge from CV, KG, and memory.
- Lets you chat directly in the terminal – type your query, get an answer, 
  and type `exit` when you’re done.

How to use:
1. Create a folder named `data/` in the same directory as this script.
2. Add your CV and related documents (PDF, Word, or text files) to the `data/` folder.
3. Make sure your `.env` file has your `OPENAI_API_KEY`.
4. Run the script with:
   ```bash
   python cv-assistant.py
"""

"""
CV Assistant with RAG + Knowledge Graph + Conversation Memory
-------------------------------------------------------------
This script builds an AI-powered CV assistant that can answer recruiter questions.

Features:
1. Uses FAISS vector store + OpenAI embeddings for semantic search (**RAG**).
2. Builds a **Domain Knowledge Graph** from your CV/docs for structured Q&A.
3. Maintains a **Conversation Memory Graph** that logs past questions and answers.
4. Automatically detects new/updated files via hashing and updates embeddings + graphs.
5. Provides an interactive chatbot interface for recruiters.

Author: Saud Ahmad
"""


import os, json, hashlib, pickle
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.chains import RetrievalQA

# NEW: KG imports
from langchain.chains import GraphQAChain
from langchain_community.graphs import NetworkxEntityGraph
import networkx as nx


# =========================
# Helpers
# =========================
def file_hash(file_path):
    """Return SHA256 hash of file contents"""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def load_document(file_path):
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


# =========================
# Setup
# =========================
print("⚙️ Loading environment variables...")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

data_dir = Path("data")
faiss_path = data_dir / "faiss_index"
hash_store_path = data_dir / "embedded_files.json"
kg_path = data_dir / "knowledge_graph.pkl"        # Domain KG
conv_memory_path = data_dir / "conversation_memory.pkl"  # Conversation KG

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load or create hash store
if hash_store_path.exists():
    with open(hash_store_path, "r") as f:
        embedded_files = json.load(f)
else:
    embedded_files = {}

# Load or create FAISS index
if faiss_path.exists():
    print("📂 Found existing FAISS index, loading it...")
    vectorstore = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
else:
    print("🆕 No FAISS index found. Creating new one...")
    vectorstore = None


# =========================
# Process files
# =========================
new_docs = []
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

for file in data_dir.glob("*"):
    if file.suffix.lower() in [".pdf", ".txt", ".doc", ".docx"]:
        h = file_hash(file)

        # New file
        if file.name not in embedded_files:
            print(f"➕ New file detected: {file.name}")
            docs = load_document(file)
            chunks = splitter.split_documents(docs)
            new_docs.extend(chunks)
            embedded_files[file.name] = h

        # File updated
        elif embedded_files[file.name] != h:
            print(f"🔄 File updated: {file.name}, re-embedding...")
            docs = load_document(file)
            chunks = splitter.split_documents(docs)
            new_docs.extend(chunks)
            embedded_files[file.name] = h

        # File unchanged
        else:
            print(f"⏩ Skipping unchanged file: {file.name}")


# =========================
# Update FAISS + Save JSON
# =========================
if new_docs:
    if vectorstore:
        vectorstore.add_documents(new_docs)
    else:
        vectorstore = FAISS.from_documents(new_docs, embeddings)

    vectorstore.save_local(str(faiss_path))
    print("✅ FAISS index updated with new/changed files.")
else:
    print("✅ No new or updated files. FAISS index unchanged.")

# Always save the latest hashes
with open(hash_store_path, "w") as f:
    json.dump(embedded_files, f, indent=2)


# =========================
# Build / Update Domain KG
# =========================
print("🧩 Building Knowledge Graph (domain)...")
graph = nx.DiGraph()

if new_docs:
    for doc in new_docs:
        text = doc.page_content
        words = text.split()
        if len(words) >= 3:
            graph.add_edge(words[0], words[1], relation="related_to")

# Save Domain KG
with open(kg_path, "wb") as f:
    pickle.dump(graph, f)
print("✅ Domain Knowledge Graph built/updated.")

# Load Domain KG
with open(kg_path, "rb") as f:
    graph = pickle.load(f)

kg = NetworkxEntityGraph(graph=graph)


# =========================
# Load / Init Conversation Memory KG
# =========================
if conv_memory_path.exists():
    with open(conv_memory_path, "rb") as f:
        conversation_graph = pickle.load(f)
    print("💬 Loaded existing Conversation Memory KG.")
else:
    conversation_graph = nx.DiGraph()
    print("🆕 Created new Conversation Memory KG.")


# =========================
# QA Chains
# =========================
print("🤖 Initializing retrievers and LLM...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# RAG QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# KG QA
kg_chain = GraphQAChain.from_llm(llm=llm, graph=kg, verbose=False)

print("✅ QA chains (RAG + KG) are ready.")

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

    # =========================
    # Update Conversation Memory KG
    # =========================
    q_node = f"Q: {query}"
    a_node = f"A: {result}"

    conversation_graph.add_node(q_node, type="question")
    conversation_graph.add_node(a_node, type="answer")
    conversation_graph.add_edge(q_node, a_node, relation="answered_by")

    # Save after each interaction
    with open(conv_memory_path, "wb") as f:
        pickle.dump(conversation_graph, f)

    print("📝 Conversation Memory KG updated.")