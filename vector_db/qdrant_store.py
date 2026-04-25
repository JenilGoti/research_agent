from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
from qdrant_client.models import VectorParams, Distance

def ensure_collection(client, collection_name: str, vector_size: int = 384):

    collections = client.get_collections().collections
    existing = [c.name for c in collections]

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"[QDRANT] Created collection: {collection_name}")


def get_vector_store():

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),      # 🔥 cloud URL
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    ensure_collection(client, "research_docs")
    vector_store = Qdrant(
        client=client,
        collection_name="research_docs",
        embeddings=embeddings,
    )

    return vector_store
