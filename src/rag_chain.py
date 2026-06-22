from openai import OpenAI


def format_sources(contexts: list[dict]) -> str:
    labels: list[str] = []
    for item in contexts:
        page_label = f" 第 {item['page']} 页" if item["page"] else ""
        labels.append(f"{item['source']}{page_label}")
    return "；".join(dict.fromkeys(labels)) or "无"


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[片段 {idx} | 来源：{item['source']}"
        f"{f' 第 {item['page']} 页' if item['page'] else ''}]\n{item['text']}"
        for idx, item in enumerate(contexts, start=1)
    )

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
