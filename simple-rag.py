import os
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
def load_document(file_path: Path):
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


def process_files(data_dir: Path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = []
    for file in data_dir.glob("*"):
        if file.suffix.lower() not in [".pdf", ".txt", ".doc", ".docx"]:
            continue
        print(f"📄 Loading: {file.name}")
        loaded = load_document(file)
        chunks = splitter.split_documents(loaded)
        docs.extend(chunks)
    return docs


# =========================
# Main
# =========================
def main():
    # Load API key
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    data_dir = Path("data")
    faiss_path = data_dir / "faiss_index"

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-large")

    # Build or load vectorstore
    if faiss_path.exists():
        print("📂 Found existing FAISS index, loading it...")
        vectorstore = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
    else:
        print("🆕 Creating FAISS index from documents...")
        docs = process_files(data_dir)
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(str(faiss_path))

    # QA chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 33})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    # Interactive Q&A loop
    print("\n🚀 CV Assistant Ready! Ask me questions (type 'exit' to quit)\n")
    while True:
        query = input("Query: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting assistant. Goodbye!")
            break

        print("💭 Thinking...")
        result = qa_chain.run(query)
        print("\nAnswer:", result)


if __name__ == "__main__":
    main()