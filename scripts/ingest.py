from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = Path("data/sample_docs")
INDEX_DIR = "data/faiss_index"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def main():
    docs = []
    for path in DATA_DIR.glob("*.txt"):
        docs.extend(TextLoader(str(path)).load())

    if not docs:
        raise ValueError("No documents found in data/sample_docs")

    db = FAISS.from_documents(docs, get_embeddings())
    db.save_local(INDEX_DIR)
    print(f"Saved FAISS index to {INDEX_DIR}")


if __name__ == "__main__":
    main()