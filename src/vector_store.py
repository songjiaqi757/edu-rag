import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.config import TOP_K, VECTOR_INDEX_PATH, VECTOR_METADATA_PATH
from src.document_loader import Chunk
from src.embeddings import embed_texts


def build_faiss_index(embeddings: np.ndarray) -> Any:
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def build_vector_store(chunks: list[Chunk]) -> Any:
    embeddings = embed_texts([chunk.text for chunk in chunks])
    return build_faiss_index(embeddings)


def retrieve_top_k(index: Any, chunks: list[Chunk], question: str, top_k: int = TOP_K) -> list[dict]:
    question_embedding = embed_texts([question])
    scores, indices = index.search(question_embedding, top_k)

    results: list[dict] = []
    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index == -1:
            continue
        chunk = chunks[chunk_index]
        results.append(
            {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "score": float(score),
            }
        )
    return results


def save_vector_store(
    index: Any,
    chunks: list[Chunk],
    index_path: Path = VECTOR_INDEX_PATH,
    metadata_path: Path = VECTOR_METADATA_PATH,
) -> bool:
    try:
        import faiss

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        with metadata_path.open("wb") as file:
            pickle.dump(chunks, file)
        return True
    except (OSError, pickle.PickleError, RuntimeError):
        return False


def load_vector_store(
    index_path: Path = VECTOR_INDEX_PATH,
    metadata_path: Path = VECTOR_METADATA_PATH,
) -> tuple[Any | None, list[Chunk]]:
    try:
        import faiss

        if not index_path.exists() or not metadata_path.exists():
            return None, []
        index = faiss.read_index(str(index_path))
        with metadata_path.open("rb") as file:
            chunks = pickle.load(file)
        return index, chunks
    except (OSError, pickle.PickleError, RuntimeError):
        return None, []


def vector_store_exists(
    index_path: Path = VECTOR_INDEX_PATH,
    metadata_path: Path = VECTOR_METADATA_PATH,
) -> bool:
    return index_path.exists() and metadata_path.exists()
