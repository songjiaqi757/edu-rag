from collections import Counter

from src.config import SIMILARITY_THRESHOLD, STOPWORDS


def get_knowledge_gaps(logs: list[dict], threshold: float = SIMILARITY_THRESHOLD) -> list[dict]:
    """从问答日志中识别课程知识库覆盖不足的问题。"""
    gaps: list[dict] = []
    for log in logs:
        is_answered = bool(log.get("is_answered", 1))
        top_score = float(log.get("top_score") or 0)
        if is_answered and top_score >= threshold:
            continue
        reason = "未命中" if not is_answered else "低相似度"
        gaps.append(
            {
                "question": log.get("question", ""),
                "top_score": top_score,
                "created_at": log.get("created_at", ""),
                "reason": reason,
            }
        )
    return gaps


def summarize_gap_keywords(gaps: list[dict], top_n: int = 10) -> list[tuple[str, int]]:
    """统计知识缺口问题中的高频关键词。"""
    import jieba

    counter: Counter[str] = Counter()
    for gap in gaps:
        for word in jieba.lcut(gap.get("question", "")):
            token = word.strip().lower()
            if len(token) < 2 or token.isdigit() or token in STOPWORDS:
                continue
            counter[token] += 1
    return counter.most_common(top_n)


def split_gap_rows(gaps: list[dict]) -> tuple[list[dict], list[dict]]:
    unanswered = [gap for gap in gaps if gap["reason"] == "未命中"]
    low_similarity = [gap for gap in gaps if gap["reason"] == "低相似度"]
    return unanswered, low_similarity
