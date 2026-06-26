import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import EVAL_QUESTIONS_PATH, TOP_K
from src.vector_store import load_vector_store, retrieve_top_k


def load_questions() -> list[dict]:
    with EVAL_QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def contains_expected_keyword(contexts: list[dict], keywords: list[str]) -> bool:
    joined = "\n".join(item.get("text", "") for item in contexts)
    return any(keyword in joined for keyword in keywords)


def main() -> None:
    index, chunks = load_vector_store()
    if index is None or not chunks:
        print("未发现已构建的 FAISS 知识库，请先运行系统并使用示例资料构建知识库。")
        return

    questions = load_questions()
    hit_count = 0
    scores: list[float] = []
    for item in questions:
        contexts = retrieve_top_k(index, chunks, item["question"], TOP_K)
        top_score = max((context["score"] for context in contexts), default=0)
        scores.append(top_score)
        if contains_expected_keyword(contexts, item["expected_keyword"]):
            hit_count += 1

    total = len(questions)
    avg_score = sum(scores) / total if total else 0
    print(f"问题总数: {total}")
    print(f"Top-3 命中数: {hit_count}")
    print(f"Top-3 命中率: {hit_count / total:.2%}" if total else "Top-3 命中率: 0.00%")
    print(f"平均最高相似度: {avg_score:.3f}")


if __name__ == "__main__":
    main()
