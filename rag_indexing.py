import os
import uuid
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from retrievers import PineconeRetriever


def read_text_files(folder: str) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith((".txt", ".md")):
                path = os.path.join(root, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        docs.append({"path": path, "text": f.read()})
                except Exception:
                    pass
    return docs


def ingest_folder(collection: str, folder: str, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
    retr = PineconeRetriever()
    texts = read_text_files(folder)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for d in texts:
        chunks = splitter.split_text(d["text"])
        for ch in chunks:
            retr.upsert_text(collection=collection, payload_id=str(uuid.uuid4()), text=ch, metadata={"source": d["path"], "collection": collection})


def ingest_all() -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    q_folder = os.path.join(base, "data", "rag_questions")
    seo_folder = os.path.join(base, "data", "rag_seo")

    if os.path.isdir(q_folder):
        ingest_folder("rag_questions", q_folder, chunk_size=300, chunk_overlap=60)
    if os.path.isdir(seo_folder):
        ingest_folder("rag_seo", seo_folder, chunk_size=800, chunk_overlap=120)


if __name__ == "__main__":
    ingest_all()


