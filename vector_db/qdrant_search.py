from qdrant_client import QdrantClient
import os
from langchain_community.embeddings import HuggingFaceEmbeddings

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def search_qdrant(query_vector, limit=4):
    results = client.query_points(
        collection_name="research_docs",
        query=query_vector,
        limit=limit,
        with_payload=True
    ).points

    return [
        {
            "text": r.payload.get("page_content", ""),
            "score": r.score
        }
        for r in results
    ]


def search_by_text(query: str, limit=5):
    print(query)
    query_vector = embeddings.embed_query(query)
    return search_qdrant(query_vector, limit)