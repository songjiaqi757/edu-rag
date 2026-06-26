import json
from datetime import datetime

from src.config import KB_VERSION_PATH


def load_kb_versions() -> list[dict]:
    """读取知识库版本记录。"""
    try:
        if not KB_VERSION_PATH.exists():
            return []
        with KB_VERSION_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def record_kb_version(
    version_name: str,
    document_count: int,
    chunk_count: int,
    note: str = "",
) -> None:
    """记录一次课程知识库构建版本。"""
    versions = load_kb_versions()
    versions.append(
        {
            "version_name": version_name,
            "document_count": document_count,
            "chunk_count": chunk_count,
            "note": note or "教师上传课程资料后构建知识库",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    KB_VERSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with KB_VERSION_PATH.open("w", encoding="utf-8") as file:
        json.dump(versions, file, ensure_ascii=False, indent=2)


def next_version_name() -> str:
    return f"v{len(load_kb_versions()) + 1}"
