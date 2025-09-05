import os
from typing import Any, Dict, Optional

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec


class PineconeRetriever:
    """
    Pinecone-based retriever using namespaces as logical collections (e.g., 'rag_questions', 'rag_seo').
    Uses a single embedding model; vector dimension must match the model.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("PINECONE_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("PINECONE_API_KEY is required")
        self.index_name = os.getenv("PINECONE_INDEX", "seo-rag")
        self.region = os.getenv("PINECONE_REGION", "us-east-1")

        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model, api_key=os.getenv("OPENAI_API_KEY"))
        # text-embedding-3-small -> 1536, text-embedding-3-large -> 3072
        self.vector_size = int(os.getenv("EMBEDDING_DIM", "1536"))

        self.pc = Pinecone(api_key=self.api_key)
        existing = {idx.name for idx in self.pc.list_indexes()}
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.vector_size,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.region),
            )
        self.index = self.pc.Index(self.index_name)

    def upsert_text(self, collection: str, payload_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        vec = self.embeddings.embed_query(text)
        md = {"text": text}
        if metadata:
            md.update(metadata)
        self.index.upsert(vectors=[{"id": payload_id, "values": vec, "metadata": md}], namespace=collection)

    def search(self, collection: str, query: str, k: int = 6, filters: Optional[Dict[str, Any]] = None):
        q_vec = self.embeddings.embed_query(query)
        resp = self.index.query(namespace=collection, vector=q_vec, top_k=k, include_values=False, include_metadata=True)
        matches = getattr(resp, "matches", []) or []
        return [
            {
                "id": m.id,
                "score": m.score,
                "text": (m.metadata or {}).get("text", ""),
                "metadata": m.metadata or {},
            }
            for m in matches
        ]


def get_retriever(kind: str) -> PineconeRetriever:
    # Single retriever instance works for any namespace via `collection` argument
    return PineconeRetriever()


