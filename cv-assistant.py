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
1. Put your documents in the `data/` folder.
2. Make sure your `.env` file has your `OPENAI_API_KEY`.
3. Run the script with:
   ```bash
   python cv-assistant.py
"""

import os, json, hashlib
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.chains import RetrievalQA


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

# Always save the latest hashes, even if no new docs
with open(hash_store_path, "w") as f:
    json.dump(embedded_files, f, indent=2)


# =========================
# QA Chain
# =========================
print("🤖 Initializing retriever and LLM...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
print("✅ QA chain is ready.")

print("\n🚀 CV Assistant Ready! Ask me questions (type 'exit' to quit)")

while True:
    query = input("\nQuery: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting assistant. Goodbye!")
        break

    print("💭 Thinking...")
    result = qa_chain.run(query)
    print("\nAnswer:", result)