from openai import APIConnectionError, APIStatusError, AuthenticationError, BadRequestError, NotFoundError, OpenAI, OpenAIError, RateLimitError


def format_sources(contexts: list[dict]) -> str:
    labels: list[str] = []
    for item in contexts:
        page_label = f" 第 {item['page']} 页" if item["page"] else ""
        labels.append(f"{item['source']}{page_label}")
    return "；".join(dict.fromkeys(labels)) or "无"


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_parts: list[str] = []
    for idx, item in enumerate(contexts, start=1):
        source = item.get("source", "未知来源")
        page = item.get("page")
        page_label = f" 第 {page} 页" if page else ""
        text = item.get("text", "")
        context_parts.append(f"[片段 {idx} | 来源：{source}{page_label}]\n{text}")
    context_text = "\n\n".join(context_parts)

    return f"""你是一个智慧教育课程助教。请根据以下课程资料回答学生问题。
要求：
1. 只依据给定资料回答；
2. 如果资料中没有答案，请说明“课程资料中暂未找到相关内容”；
3. 回答要清晰、适合学生理解；
4. 最后列出参考片段来源。

课程资料：
{context_text}

学生问题：
{question}
"""


def generate_answer(
    api_key: str,
    model_name: str,
    question: str,
    contexts: list[dict],
) -> tuple[str, str | None]:
    """调用大语言模型生成回答，失败时返回友好错误信息。"""
    try:
        client = OpenAI(api_key=api_key)
        prompt = build_prompt(question, contexts)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个智慧教育课程助教。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or "", None
    except Exception as error:
        return "", format_openai_error(error)


def generate_demo_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return "课程资料中暂未找到相关内容。"

    lines = [
        "当前处于演示模式，未调用大语言模型。",
        "",
        "根据课程资料，检索到以下可能相关内容：",
    ]
    for idx, item in enumerate(contexts[:3], start=1):
        text = " ".join(item["text"].split())
        excerpt = text[:220] + ("..." if len(text) > 220 else "")
        page_label = f" 第 {item['page']} 页" if item["page"] else ""
        lines.append(f"片段 {idx}（来源：{item['source']}{page_label}）：{excerpt}")

    lines.extend(["", f"学生问题：{question}", "建议结合上述资料继续学习，或向教师进一步确认。"])
    return "\n".join(lines)


def format_openai_error(error: Exception) -> str:
    if isinstance(error, AuthenticationError):
        return "OpenAI API Key 无效或权限不足，请检查 Key 是否填写正确。"
    if isinstance(error, APIConnectionError):
        return "连接 OpenAI 服务失败，请检查网络连接后重试。"
    if isinstance(error, NotFoundError):
        return "OpenAI 模型名可能不存在，请检查模型名称。"
    if isinstance(error, BadRequestError):
        return "OpenAI 请求参数有误，请检查模型名或输入内容。"
    if isinstance(error, RateLimitError):
        return "OpenAI API 额度不足或请求过于频繁，请稍后重试或检查账户额度。"
    if isinstance(error, APIStatusError):
        return f"OpenAI 服务返回错误状态码 {error.status_code}，系统已回退到演示模式回答。"
    if isinstance(error, OpenAIError):
        return "OpenAI 服务调用失败，系统已回退到演示模式回答。"
    return "大模型调用失败，系统已回退到演示模式回答。"
