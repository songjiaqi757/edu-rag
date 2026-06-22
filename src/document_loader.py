import os
import tempfile
from dataclasses import dataclass
from typing import BinaryIO, Iterable


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None


def read_pdf(file: BinaryIO) -> list[Chunk]:
    import fitz

    chunks: list[Chunk] = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                chunks.append(Chunk(text=text, source=file.name, page=page_index))
        doc.close()
    finally:
        os.unlink(tmp_path)

    return chunks


def read_txt(file: BinaryIO) -> list[Chunk]:
    raw = file.getvalue()
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            text = raw.decode(encoding).strip()
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="ignore").strip()

    return [Chunk(text=text, source=file.name)] if text else []


def load_documents(uploaded_files: Iterable[BinaryIO]) -> list[Chunk]:
    documents: list[Chunk] = []
    for file in uploaded_files:
        file_type = file.name.rsplit(".", 1)[-1].lower()
        if file_type == "pdf":
            documents.extend(read_pdf(file))
        elif file_type == "txt":
            documents.extend(read_txt(file))
    return documents
