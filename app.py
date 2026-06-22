import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

import faiss
import fitz
import jieba
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.database import get_qa_logs, init_db, save_qa_log


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
TOP_K = 3
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


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None


def read_pdf(file) -> list[Chunk]:
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


def read_txt(file) -> list[Chunk]:
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


def load_documents(uploaded_files) -> list[Chunk]:
    documents: list[Chunk] = []
    for file in uploaded_files:
        file_type = file.name.rsplit(".", 1)[-1].lower()
        if file_type == "pdf":
            documents.extend(read_pdf(file))
        elif file_type == "txt":
            documents.extend(read_txt(file))
    return documents


def split_documents(documents: Iterable[Chunk]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )

    split_chunks: list[Chunk] = []
    for doc in documents:
        for text in splitter.split_text(doc.text):
            cleaned = text.strip()
            if cleaned:
                split_chunks.append(
                    Chunk(text=cleaned, source=doc.source, page=doc.page)
                )
    return split_chunks


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_top_k(question: str, top_k: int = TOP_K) -> list[dict]:
    model = get_embedding_model()
    question_embedding = embed_texts(model, [question])
    scores, indices = st.session_state.index.search(question_embedding, top_k)

    results = []
    for score, index in zip(scores[0], indices[0]):
        if index == -1:
            continue
        chunk = st.session_state.chunks[index]
        results.append(
            {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "score": float(score),
            }
        )
    return results


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[片段 {idx} | 来源：{item['source']}"
        f"{f' 第 {item['page']} 页' if item['page'] else ''}]\n{item['text']}"
        for idx, item in enumerate(contexts, start=1)
    )

    return f"""你是一个严谨的课程助教。请只根据给定课程资料回答学生问题。
如果资料中没有足够信息，请明确说明“课程资料中没有找到足够依据”。
回答要清晰、准确，并尽量使用中文。

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
            {"role": "system", "content": "你是一个面向学生的课程知识库问答助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def render_citations(contexts: list[dict]) -> None:
    st.subheader("参考资料片段")
    for idx, item in enumerate(contexts, start=1):
        page_label = f" · 第 {item['page']} 页" if item["page"] else ""
        with st.expander(
            f"片段 {idx} · {item['source']}{page_label} · 相似度 {item['score']:.3f}",
            expanded=idx == 1,
        ):
            st.write(item["text"])


def format_sources(contexts: list[dict]) -> str:
    labels = []
    for item in contexts:
        page_label = f" 第 {item['page']} 页" if item["page"] else ""
        labels.append(f"{item['source']}{page_label}")
    return "；".join(dict.fromkeys(labels)) or "无"


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


def plot_recent_trend(trend_rows: list[dict]):
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


def init_state() -> None:
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("index", None)
    st.session_state.setdefault("page", "课程问答")


def render_teacher_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.title("edu-rag")
        st.header("教师端")
        uploaded_files = st.file_uploader(
            "上传 PDF 或 TXT 课程资料",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
        )
        model_name = st.text_input("OpenAI 模型", value=DEFAULT_OPENAI_MODEL)

        if st.button("构建知识库", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("请先上传至少一个 PDF 或 TXT 文件。")
            else:
                with st.spinner("正在解析、分块并构建向量索引..."):
                    documents = load_documents(uploaded_files)
                    chunks = split_documents(documents)
                    if not chunks:
                        st.error("没有从文件中解析出可用文本。")
                    else:
                        embedding_model = get_embedding_model()
                        embeddings = embed_texts(
                            embedding_model, [chunk.text for chunk in chunks]
                        )
                        st.session_state.chunks = chunks
                        st.session_state.index = build_faiss_index(embeddings)
                        st.success(f"知识库构建完成，共 {len(chunks)} 个文本片段。")

        if st.session_state.index is not None:
            st.info(f"当前知识库：{len(st.session_state.chunks)} 个文本片段")
        else:
            st.warning("当前知识库为空，请先上传资料并点击“构建知识库”。")

        st.divider()
        if st.button("问答记录", use_container_width=True):
            st.session_state.page = "问答记录"
        if st.button("学情分析", use_container_width=True):
            st.session_state.page = "学情分析"

        st.divider()
        st.header("学生端")
        if st.button("课程问答", use_container_width=True):
            st.session_state.page = "课程问答"

    return api_key, model_name


def render_qa_page(api_key: str, model_name: str) -> None:
    st.title("课程知识库问答系统")

    if st.session_state.index is None:
        st.info("知识库为空。请教师先在左侧“教师端”上传 PDF/TXT 课程资料，并点击“构建知识库”。")
    else:
        st.success(f"知识库已就绪，可基于 {len(st.session_state.chunks)} 个课程片段回答问题。")

    question = st.text_area(
        "学生问题",
        placeholder="例如：这门课的核心概念是什么？",
        height=110,
    )

    ask = st.button("提交问题", type="primary")
    if ask:
        if st.session_state.index is None:
            st.warning("请先在侧边栏上传资料并构建知识库。")
        elif not api_key:
            st.warning("请填写 OpenAI API Key，或设置 OPENAI_API_KEY 环境变量。")
        elif not question.strip():
            st.warning("请输入问题。")
        else:
            with st.spinner("正在检索相关片段并生成回答..."):
                contexts = retrieve_top_k(question.strip(), TOP_K)
                answer = generate_answer(
                    api_key=api_key,
                    model_name=model_name,
                    question=question.strip(),
                    contexts=contexts,
                )

            st.subheader("回答")
            st.write(answer)
            st.subheader("参考资料来源")
            st.write(format_sources(contexts))
            render_citations(contexts)

            saved = save_qa_log(
                question=question.strip(),
                answer=answer,
                sources=format_sources(contexts),
            )
            if not saved:
                st.warning("回答已生成，但问答日志保存失败。")


def render_logs_page() -> None:
    st.title("问答记录")
    logs = get_qa_logs()

    if not logs:
        st.info("暂无问答记录。")
        return

    rows = [
        {
            "问题": log["question"],
            "回答摘要": summarize_answer(log["answer"]),
            "来源": log["sources"],
            "时间": log["created_at"],
        }
        for log in logs
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_learning_analytics_page() -> None:
    st.title("学情分析")
    logs = get_qa_logs()

    total_questions = len(logs)
    st.metric("总提问数", total_questions)

    if not logs:
        st.info("暂无历史提问，生成问答后这里会显示学情分析。")
        return

    st.subheader("最近 7 天提问趋势")
    trend_rows = get_recent_7_day_counts(logs)
    fig = plot_recent_trend(trend_rows)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("高频关键词")
    keywords = extract_keywords(logs)
    if keywords:
        keyword_rows = [{"关键词": word, "次数": count} for word, count in keywords]
        st.dataframe(keyword_rows, use_container_width=True, hide_index=True)
    else:
        st.info("暂无可统计的关键词。")

    st.subheader("最近问题")
    recent_rows = [
        {
            "问题": log["question"],
            "来源": log["sources"],
            "时间": log["created_at"],
        }
        for log in logs[:10]
    ]
    st.dataframe(recent_rows, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="课程知识库问答系统", page_icon="📚", layout="wide")
    init_state()
    init_db()

    api_key, model_name = render_teacher_sidebar()

    if st.session_state.page == "课程问答":
        render_qa_page(api_key, model_name)
    elif st.session_state.page == "问答记录":
        render_logs_page()
    else:
        render_learning_analytics_page()


if __name__ == "__main__":
    main()
