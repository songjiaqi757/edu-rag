import os

import streamlit as st

from src.analytics import build_log_rows, build_problem_rows, build_recent_rows
from src.analytics import extract_difficult_keywords, extract_keywords
from src.analytics import get_low_similarity_logs, get_recent_7_day_counts, get_unanswered_logs
from src.analytics import plot_recent_trend, summarize_weak_points
from src.config import DEFAULT_OPENAI_MODEL, SIMILARITY_THRESHOLD, TOP_K
from src.database import get_qa_logs, init_db, save_qa_log
from src.document_loader import load_documents
from src.evaluation import compute_coverage_metrics
from src.gap_analysis import get_knowledge_gaps, split_gap_rows, summarize_gap_keywords
from src.kb_version import load_kb_versions, next_version_name, record_kb_version
from src.rag_chain import format_sources, generate_answer, generate_demo_answer
from src.splitter import split_documents
from src.ui import apply_theme, question_input, render_citations, render_hero, status_message
from src.vector_store import build_vector_store, load_vector_store, retrieve_top_k
from src.vector_store import save_vector_store, vector_store_exists


def init_state() -> None:
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("index", None)
    st.session_state.setdefault("page", "课程问答")
    st.session_state.setdefault("question_input", "")


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
        record_kb_version(
            next_version_name(),
            document_count=len(uploaded_files),
            chunk_count=len(chunks),
            note="教师上传课程资料后构建知识库",
        )
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
        st.title("EduRAG")
        st.caption("面向课堂资料的智慧课程助教")
        st.header("教师端 · 智慧教学")
        uploaded_files = st.file_uploader("上传 PDF 或 TXT 课程资料", type=["pdf", "txt"], accept_multiple_files=True)
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        model_name = st.text_input("OpenAI 模型", value=DEFAULT_OPENAI_MODEL)

        if st.button("构建知识库", type="primary", use_container_width=True):
            build_knowledge_base(uploaded_files)

        if st.button("加载已有知识库", use_container_width=True):
            load_existing_knowledge_base()

        status = f"{len(st.session_state.chunks)} 个片段" if st.session_state.index is not None else "待构建"
        st.metric("课程知识库", status)

        st.divider()
        if st.button("问答记录", use_container_width=True):
            st.session_state.page = "问答记录"
        if st.button("学情分析", use_container_width=True):
            st.session_state.page = "学情分析"
        if st.button("知识缺口", use_container_width=True):
            st.session_state.page = "知识缺口"
        if st.button("知识库版本", use_container_width=True):
            st.session_state.page = "知识库版本"

        st.divider()
        st.header("学生端 · 课程助教")
        if st.button("课程问答", use_container_width=True):
            st.session_state.page = "课程问答"

    return api_key, model_name


def render_qa_page(api_key: str, model_name: str) -> None:
    render_hero(
        "智慧课程助教问答",
        "围绕教师上传的课程资料进行检索增强回答，帮助学生获得有依据、可追溯的学习支持。",
        ["课程资料优先", "Top-3 片段检索", f"命中阈值 {SIMILARITY_THRESHOLD:.2f}"],
    )

    ready = st.session_state.index is not None
    status_message(ready, len(st.session_state.chunks))
    c1, c2, c3 = st.columns(3)
    c1.metric("知识库状态", "已就绪" if ready else "未构建")
    c2.metric("课程片段", len(st.session_state.chunks))
    c3.metric("检索数量", f"Top-{TOP_K}")

    question = question_input()

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
        demo_mode = False
        api_error = ""
        if not is_answered:
            answer = "课程资料中暂未找到相关内容。建议向教师提问，或请教师补充相关课程资料。"
        elif not api_key:
            answer = generate_demo_answer(clean_question, contexts)
            demo_mode = True
        else:
            answer, api_error = generate_answer(api_key, model_name, clean_question, contexts)
            if api_error:
                answer = generate_demo_answer(clean_question, contexts)
                demo_mode = True

    sources = format_sources(contexts)
    st.markdown('<div class="edu-answer">', unsafe_allow_html=True)
    st.subheader("课程助教回答")
    st.write(answer)
    st.markdown("</div>", unsafe_allow_html=True)
    st.subheader("参考资料来源")
    st.info(sources)
    render_citations(contexts)

    if demo_mode and not api_error:
        st.warning("当前使用演示模式，未调用大模型。")
    if demo_mode and api_error:
        st.warning("当前展示演示模式回答。")
    if api_error:
        st.error(api_error)
    if not is_answered:
        st.info(f"最高检索相似度为 {top_score:.3f}，低于课程助教阈值 {SIMILARITY_THRESHOLD:.2f}。")

    if not save_qa_log(clean_question, answer, sources, is_answered, top_score, sources):
        st.warning("回答已生成，但问答日志保存失败。")


def render_logs_page() -> None:
    render_hero("课程助教问答记录", "沉淀学生真实提问，为教师复盘课堂理解情况提供依据。", ["自动保存", "来源追踪", "命中判定"])
    logs = get_qa_logs()
    if not logs:
        st.info("暂无问答记录。")
        return

    st.dataframe(build_log_rows(logs), use_container_width=True, hide_index=True)


def render_learning_analytics_page() -> None:
    render_hero("智慧教学学情分析", "从问答日志中识别未命中问题、疑难关键词和可能薄弱知识点。", ["学习诊断", "疑难聚类", "资料补全参考"])
    logs = get_qa_logs()
    unanswered = get_unanswered_logs(logs)
    low_similarity = get_low_similarity_logs(logs, SIMILARITY_THRESHOLD)
    metrics = compute_coverage_metrics(logs)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("总提问数", len(logs))
    col2.metric("已回答", metrics["answered"])
    col3.metric("未命中", metrics["unanswered"])
    col4.metric("覆盖率", f"{metrics['coverage_rate']:.0%}")
    col5.metric("平均相似度", f"{metrics['avg_top_score']:.3f}")
    st.caption("课程问答覆盖率反映当前知识库对学生问题的支撑程度。覆盖率较低时，教师可以根据知识缺口页面补充课程资料。")

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


def render_gap_page() -> None:
    render_hero("知识缺口识别", "识别课程知识库中覆盖不足的内容，支持教师补充资料并迭代更新。", ["未命中问题", "低相似度问题", "教师反馈闭环"])
    logs = get_qa_logs()
    gaps = get_knowledge_gaps(logs, SIMILARITY_THRESHOLD)
    unanswered, low_similarity = split_gap_rows(gaps)
    st.metric("知识缺口总数", len(gaps))
    st.info("以下问题反映了当前课程知识库中可能覆盖不足的内容。教师可以根据这些问题补充课程资料，并重新构建知识库，以提升后续问答覆盖率。")
    if not gaps:
        st.success("当前暂未识别到明显知识缺口。")
        return
    st.subheader("未命中问题列表")
    st.dataframe(unanswered, use_container_width=True, hide_index=True)
    st.subheader("低相似度问题列表")
    st.dataframe(low_similarity, use_container_width=True, hide_index=True)
    st.subheader("高频疑难关键词")
    st.dataframe(
        [{"关键词": word, "次数": count} for word, count in summarize_gap_keywords(gaps)],
        use_container_width=True,
        hide_index=True,
    )


def render_versions_page() -> None:
    render_hero("知识库版本管理", "记录教师补充资料和重新构建知识库的过程，体现课程知识库迭代闭环。", ["版本记录", "构建时间", "资料迭代"])
    versions = load_kb_versions()
    if not versions:
        st.info("暂无知识库版本记录。构建知识库后会自动生成版本记录。")
        return
    st.dataframe(versions[::-1], use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="课程知识库问答系统", page_icon="📚", layout="wide")
    apply_theme()
    init_state()
    init_db()

    api_key, model_name = render_teacher_sidebar()

    if st.session_state.page == "课程问答":
        render_qa_page(api_key, model_name)
    elif st.session_state.page == "学情分析":
        render_learning_analytics_page()
    elif st.session_state.page == "知识缺口":
        render_gap_page()
    elif st.session_state.page == "知识库版本":
        render_versions_page()
    else:
        render_logs_page()


if __name__ == "__main__":
    main()
