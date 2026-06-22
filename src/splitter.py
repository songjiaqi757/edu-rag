from collections.abc import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, TEXT_SEPARATORS
from src.document_loader import Chunk


def split_documents(documents: Iterable[Chunk]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=TEXT_SEPARATORS,
    )

    split_chunks: list[Chunk] = []
    for document in documents:
        for text in splitter.split_text(document.text):
            cleaned = text.strip()
            if cleaned:
                split_chunks.append(
                    Chunk(text=cleaned, source=document.source, page=document.page)
                )
    return split_chunks
