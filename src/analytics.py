from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from src.config import STOPWORDS


def summarize_answer(answer: str, max_length: int = 120) -> str:
    one_line = " ".join(answer.split())
    if len(one_line) <= max_length:
        return one_line
    return f"{one_line[:max_length]}..."


def parse_created_at(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def get_recent_7_day_counts(logs: list[dict]) -> list[dict]:
    today = datetime.now().date()
    days = [today - timedelta(days=offset) for offset in range(6, -1, -1)]
    counts = {day: 0 for day in days}

    for log in logs:
        created_at = parse_created_at(log.get("created_at", ""))
        if not created_at:
            continue
        day = created_at.date()
        if day in counts:
            counts[day] += 1

    return [{"date": day.strftime("%m-%d"), "count": counts[day]} for day in days]


def extract_keywords(logs: list[dict], top_n: int = 15) -> list[tuple[str, int]]:
    import jieba

    counter: Counter[str] = Counter()
    for log in logs:
        question = log.get("question", "")
        for word in jieba.lcut(question):
            token = word.strip().lower()
            if len(token) < 2 or token in STOPWORDS:
                continue
            if token.isdigit():
                continue
            counter[token] += 1
    return counter.most_common(top_n)


def get_unanswered_logs(logs: list[dict]) -> list[dict]:
    return [log for log in logs if not bool(log.get("is_answered", 1))]


def get_low_similarity_logs(logs: list[dict], threshold: float) -> list[dict]:
    return [log for log in logs if _is_low_similarity(log, threshold)]


def extract_difficult_keywords(logs: list[dict], threshold: float) -> list[tuple[str, int]]:
    difficult_logs = get_unanswered_logs(logs) or get_low_similarity_logs(logs, threshold)
    return extract_keywords(difficult_logs, top_n=10)


def summarize_weak_points(keywords: list[tuple[str, int]]) -> str:
    if not keywords:
        return "当前问答记录较少，暂未形成明确的薄弱知识点画像。"
    terms = "、".join(word for word, _ in keywords[:5])
    return f"学生可能在“{terms}”相关知识点上存在理解盲区，建议教师补充讲解、增加案例或完善课程资料。"


def build_log_rows(logs: list[dict]) -> list[dict]:
    return [
        {
            "问题": log["question"],
            "回答摘要": summarize_answer(log["answer"]),
            "是否命中": "是" if log.get("is_answered") else "否",
            "最高相似度": f"{float(log.get('top_score') or 0):.3f}",
            "课程来源": log.get("course_sources") or log["sources"],
            "时间": log["created_at"],
        }
        for log in logs
    ]


def build_problem_rows(logs: list[dict], limit: int = 10) -> list[dict]:
    return [
        {
            "问题": log["question"],
            "最高相似度": f"{float(log.get('top_score') or 0):.3f}",
            "课程来源": log.get("course_sources") or log.get("sources", ""),
            "时间": log["created_at"],
        }
        for log in logs[:limit]
    ]


def build_recent_rows(logs: list[dict], limit: int = 10) -> list[dict]:
    return [
        {
            "问题": log["question"],
            "课程来源": log.get("course_sources") or log["sources"],
            "时间": log["created_at"],
        }
        for log in logs[:limit]
    ]


def _is_low_similarity(log: dict, threshold: float) -> bool:
    score = float(log.get("top_score") or 0)
    return not bool(log.get("is_answered", 1)) or 0 < score < threshold


def plot_recent_trend(trend_rows: list[dict]) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.6))
    dates = [row["date"] for row in trend_rows]
    counts = [row["count"] for row in trend_rows]

    ax.plot(dates, counts, marker="o", linewidth=2)
    ax.fill_between(dates, counts, alpha=0.12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Questions")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig
