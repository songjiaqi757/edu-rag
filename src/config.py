from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "qa_logs.db"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
VECTOR_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
VECTOR_METADATA_PATH = VECTOR_STORE_DIR / "chunks.pkl"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.35

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TEXT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

STOPWORDS = {
    "什么",
    "怎么",
    "如何",
    "为什么",
    "这个",
    "那个",
    "一个",
    "一下",
    "可以",
    "请问",
    "请",
    "吗",
    "呢",
    "的",
    "了",
    "和",
    "与",
    "是",
    "在",
    "对",
    "中",
    "有",
    "及",
    "或",
}
