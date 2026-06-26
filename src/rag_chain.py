from openai import APIConnectionError, AuthenticationError, BadRequestError, NotFoundError, OpenAI, OpenAIError


def format_sources(contexts: list[dict]) -> str:
    labels: list[str] = []
    for item in contexts:
        page_label = f" 第 {item['page']} 页" if item["page"] else ""
        labels.append(f"{item['source']}{page_label}")
    return "；".join(dict.fromkeys(labels)) or "无"


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_parts: list[str] = []
    for idx, item in enumerate(contexts, start=1):
        page_label = f" 第 {item['page']} 页" if item["page"] else ""
        header = f"[片段 {idx} | 来源：{item['source']}{page_label}]"
        context_parts.append(f"{header}\n{item['text']}")
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


def generate_answer(api_key: str, model_name: str, question: str, contexts: list[dict]) -> str:
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
    return response.choices[0].message.content or ""


def generate_demo_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return "课程资料中暂未找到相关内容。"

    lines = [
        "当前为演示模式，以下回答根据检索到的课程片段抽取整理：",
        "",
        f"针对问题“{question}”，课程资料中相关内容主要包括：",
    ]
    for idx, item in enumerate(contexts[:3], start=1):
        text = " ".join(item["text"].split())
        excerpt = text[:220] + ("..." if len(text) > 220 else "")
        page_label = f" 第 {item['page']} 页" if item["page"] else ""
        lines.append(f"{idx}. {excerpt}（来源：{item['source']}{page_label}）")

    lines.extend(["", "建议结合上述片段继续阅读原始课程资料，必要时向教师确认细节。"])
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
    if isinstance(error, OpenAIError):
        return "OpenAI 服务调用失败，系统已回退到演示模式回答。"
    return "大模型调用失败，系统已回退到演示模式回答。"
