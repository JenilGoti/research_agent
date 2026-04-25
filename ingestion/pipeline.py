from utils.data_loader import load_all_documents
from vector_db.qdrant_store import get_vector_store
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingest_all(data_dir: str):

    # 1. Load docs
    docs = load_all_documents(data_dir)

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    # 3. Store in vector DB
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    return f"Ingested {len(chunks)} chunks"