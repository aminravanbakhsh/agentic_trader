from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "data/faiss_index"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_retriever():
    db = FAISS.load_local(
        INDEX_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    return db.as_retriever(search_kwargs={"k": 3})