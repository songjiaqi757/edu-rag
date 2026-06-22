import os

import streamlit as st

from src.analytics import build_log_rows, build_problem_rows, build_recent_rows
from src.analytics import extract_difficult_keywords, extract_keywords
from src.analytics import get_low_similarity_logs, get_recent_7_day_counts, get_unanswered_logs
from src.analytics import plot_recent_trend, summarize_weak_points
from src.config import DEFAULT_OPENAI_MODEL, SIMILARITY_THRESHOLD, TOP_K
from src.database import get_qa_logs, init_db, save_qa_log
from src.document_loader import load_documents
from src.rag_chain import format_sources, generate_answer
from src.splitter import split_documents
from src.vector_store import build_vector_store, load_vector_store, retrieve_top_k
from src.vector_store import save_vector_store, vector_store_exists


def init_state() -> None:
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("index", None)
    st.session_state.setdefault("page", "课程问答")


def build_knowledge_base(uploaded_files: list) -> None:
    if not uploaded_files:
        st.warning("请先上传至少一个 PDF 或 TXT 文件。")
        return

    with st.spinner("正在解析、分块并构建向量索引..."):
        documents = load_documents(uploaded_files)
        chunks = split_documents(documents)
        if not chunks:
            st.error("没有从文件中解析出可用文本。")
            return

        st.session_state.chunks = chunks
        st.session_state.index = build_vector_store(chunks)
        if save_vector_store(st.session_state.index, chunks):
            st.success("知识库已保存到 data/vector_store/。")
        else:
            st.warning("知识库已构建，但保存到本地失败。")
        st.success(f"知识库构建完成，共 {len(chunks)} 个文本片段。")


def load_existing_knowledge_base() -> None:
    if not vector_store_exists():
        st.warning("未发现已有知识库文件，请先上传资料并构建知识库。")
        return

    with st.spinner("正在加载已有知识库..."):
        index, chunks = load_vector_store()
        if index is None or not chunks:
            st.error("知识库文件读取失败，请重新构建知识库。")
            return

        st.session_state.index = index
        st.session_state.chunks = chunks
        st.success(f"已加载已有知识库，共 {len(chunks)} 个文本片段。")


def render_teacher_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.title("edu-rag")
        st.header("教师端 · 智慧教学")
        uploaded_files = st.file_uploader("上传 PDF 或 TXT 课程资料", type=["pdf", "txt"], accept_multiple_files=True)
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        model_name = st.text_input("OpenAI 模型", value=DEFAULT_OPENAI_MODEL)

        if st.button("构建知识库", type="primary", use_container_width=True):
            build_knowledge_base(uploaded_files)

        if st.button("加载已有知识库", use_container_width=True):
            load_existing_knowledge_base()

        if st.session_state.index is not None:
            st.info(f"当前知识库：{len(st.session_state.chunks)} 个文本片段")
        else:
            st.warning("课程知识库为空，请先上传资料并构建课程助教知识库。")

        st.divider()
        if st.button("问答记录", use_container_width=True):
            st.session_state.page = "问答记录"
        if st.button("学情分析", use_container_width=True):
            st.session_state.page = "学情分析"

        st.divider()
        st.header("学生端 · 课程助教")
        if st.button("课程问答", use_container_width=True):
            st.session_state.page = "课程问答"

    return api_key, model_name


def render_citations(contexts: list[dict]) -> None:
    st.subheader("参考资料片段")
    for idx, item in enumerate(contexts, start=1):
        page_label = f" · 第 {item['page']} 页" if item["page"] else ""
        with st.expander(
            f"片段 {idx} · {item['source']}{page_label} · 相似度 {item['score']:.3f}",
            expanded=idx == 1,
        ):
            st.write(item["text"])


def render_qa_page(api_key: str, model_name: str) -> None:
    st.title("智慧课程助教问答")

    if st.session_state.index is None:
        st.info("课程知识库为空。请教师先在左侧“教师端 · 智慧教学”上传资料并构建知识库。")
    else:
        st.success(f"课程助教已就绪，可基于 {len(st.session_state.chunks)} 个课程片段回答问题。")

    question = st.text_area("学生问题", placeholder="例如：这门课的核心概念是什么？", height=110)

    if not st.button("提交问题", type="primary"):
        return

    clean_question = question.strip()
    if st.session_state.index is None:
        st.warning("请先在侧边栏上传资料并构建知识库。")
        return
    if not clean_question:
        st.warning("请输入问题。")
        return

    with st.spinner("正在检索相关片段并生成回答..."):
        contexts = retrieve_top_k(st.session_state.index, st.session_state.chunks, clean_question, TOP_K)
        top_score = max((item["score"] for item in contexts), default=0.0)
        is_answered = top_score >= SIMILARITY_THRESHOLD
        if is_answered and api_key:
            answer = generate_answer(api_key, model_name, clean_question, contexts)
        elif is_answered:
            st.warning("请填写 OpenAI API Key，或设置 OPENAI_API_KEY 环境变量。")
            return
        else:
            answer = "课程资料中暂未找到相关内容。建议向教师提问，或请教师补充相关课程资料。"

    sources = format_sources(contexts)
    st.subheader("回答")
    st.write(answer)
    st.subheader("参考资料来源")
    st.write(sources)
    render_citations(contexts)

    if not is_answered:
        st.info(f"最高检索相似度为 {top_score:.3f}，低于课程助教阈值 {SIMILARITY_THRESHOLD:.2f}。")

    if not save_qa_log(clean_question, answer, sources, is_answered, top_score, sources):
        st.warning("回答已生成，但问答日志保存失败。")


def render_logs_page() -> None:
    st.title("课程助教问答记录")
    logs = get_qa_logs()
    if not logs:
        st.info("暂无问答记录。")
        return

    st.dataframe(build_log_rows(logs), use_container_width=True, hide_index=True)


def render_learning_analytics_page() -> None:
    st.title("智慧教学学情分析")
    logs = get_qa_logs()
    unanswered = get_unanswered_logs(logs)
    low_similarity = get_low_similarity_logs(logs, SIMILARITY_THRESHOLD)
    col1, col2, col3 = st.columns(3)
    col1.metric("总提问数", len(logs))
    col2.metric("未命中问题", len(unanswered))
    col3.metric("低相似度问题", len(low_similarity))

    if not logs:
        st.info("暂无历史提问。学生使用课程助教后，这里会生成智慧教学分析。")
        return

    st.subheader("最近 7 天提问趋势")
    fig = plot_recent_trend(get_recent_7_day_counts(logs))
    st.pyplot(fig)
    fig.clear()

    st.subheader("高频关键词")
    keywords = extract_keywords(logs)
    if keywords:
        st.dataframe(
            [{"关键词": word, "次数": count} for word, count in keywords],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("暂无可统计的关键词。")

    st.subheader("高频疑难关键词")
    difficult_keywords = extract_difficult_keywords(logs, SIMILARITY_THRESHOLD)
    if difficult_keywords:
        st.dataframe(
            [{"疑难关键词": word, "出现次数": count} for word, count in difficult_keywords],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("暂无明显疑难关键词。")

    st.subheader("学生可能薄弱知识点总结")
    st.write(summarize_weak_points(difficult_keywords))

    st.subheader("未命中问题列表")
    if unanswered:
        st.dataframe(build_problem_rows(unanswered), use_container_width=True, hide_index=True)
    else:
        st.info("暂无未命中问题。")

    st.subheader("低相似度问题列表")
    if low_similarity:
        st.dataframe(build_problem_rows(low_similarity), use_container_width=True, hide_index=True)
    else:
        st.info("暂无低相似度问题。")

    st.subheader("最近问题")
    st.dataframe(build_recent_rows(logs), use_container_width=True, hide_index=True)


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
