def compute_coverage_metrics(logs: list[dict]) -> dict:
    """计算课程问答覆盖率指标。"""
    total = len(logs)
    answered = sum(1 for log in logs if bool(log.get("is_answered", 1)))
    unanswered = total - answered
    scores = [float(log.get("top_score") or 0) for log in logs]
    avg_top_score = sum(scores) / total if total else 0
    return {
        "total": total,
        "answered": answered,
        "unanswered": unanswered,
        "coverage_rate": answered / total if total else 0,
        "avg_top_score": avg_top_score,
    }
