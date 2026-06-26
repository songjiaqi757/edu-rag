import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --edu-bg: #f6f8fb;
          --edu-panel: #ffffff;
          --edu-text: #202532;
          --edu-muted: #667085;
          --edu-line: #e6eaf0;
          --edu-blue: #2563eb;
          --edu-blue-soft: #eaf1ff;
          --edu-green: #12a150;
          --edu-red: #ef4444;
        }
        .stApp { background: var(--edu-bg); }
        .block-container {
          max-width: 1180px;
          padding-top: 3.5rem;
          padding-bottom: 4rem;
        }
        [data-testid="stSidebar"] {
          background: #ffffff;
          border-right: 1px solid var(--edu-line);
        }
        [data-testid="stSidebar"] h1 {
          color: var(--edu-text);
          font-size: 1.35rem;
          margin-bottom: .25rem;
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
          font-size: 1rem;
          color: var(--edu-text);
          margin-top: 1.25rem;
        }
        .edu-hero {
          padding: 30px 34px;
          border: 1px solid var(--edu-line);
          border-radius: 18px;
          background: linear-gradient(135deg, #ffffff 0%, #eef5ff 100%);
          margin-bottom: 22px;
        }
        .edu-eyebrow {
          color: var(--edu-blue);
          font-weight: 700;
          font-size: .86rem;
          margin-bottom: 8px;
        }
        .edu-title {
          color: var(--edu-text);
          font-size: 2.35rem;
          line-height: 1.15;
          font-weight: 800;
          margin: 0 0 10px 0;
        }
        .edu-subtitle {
          color: var(--edu-muted);
          font-size: 1.02rem;
          max-width: 760px;
        }
        .edu-pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 18px;
        }
        .edu-pill {
          border-radius: 999px;
          background: #fff;
          border: 1px solid var(--edu-line);
          color: #344054;
          padding: 7px 12px;
          font-size: .86rem;
          font-weight: 600;
        }
        .edu-panel {
          border: 1px solid var(--edu-line);
          border-radius: 16px;
          background: var(--edu-panel);
          padding: 20px 22px;
          margin-bottom: 18px;
        }
        .edu-panel-title {
          font-size: 1.05rem;
          font-weight: 800;
          color: var(--edu-text);
          margin-bottom: 8px;
        }
        .edu-muted { color: var(--edu-muted); }
        .edu-status-ok {
          color: #05603a;
          background: #ecfdf3;
          border: 1px solid #abefc6;
          border-radius: 12px;
          padding: 12px 14px;
          margin-bottom: 16px;
        }
        .edu-status-warn {
          color: #854a0e;
          background: #fffaeb;
          border: 1px solid #fedf89;
          border-radius: 12px;
          padding: 12px 14px;
          margin-bottom: 16px;
        }
        .edu-answer {
          border-left: 4px solid var(--edu-blue);
          background: #ffffff;
          border-radius: 14px;
          padding: 18px 20px;
          border-top: 1px solid var(--edu-line);
          border-right: 1px solid var(--edu-line);
          border-bottom: 1px solid var(--edu-line);
          margin-top: 14px;
        }
        div.stButton > button {
          border-radius: 10px;
          font-weight: 700;
          min-height: 2.6rem;
        }
        div[data-testid="stMetric"] {
          background: #ffffff;
          border: 1px solid var(--edu-line);
          border-radius: 14px;
          padding: 14px 16px;
        }
        textarea {
          border-radius: 14px !important;
          border-color: #d0d5dd !important;
          background: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, badges: list[str]) -> None:
    pills = "".join(f'<span class="edu-pill">{badge}</span>' for badge in badges)
    st.markdown(
        f"""
        <div class="edu-hero">
          <div class="edu-eyebrow">EduRAG 智慧教育工作台</div>
          <h1 class="edu-title">{title}</h1>
          <div class="edu-subtitle">{subtitle}</div>
          <div class="edu-pill-row">{pills}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_message(ready: bool, chunk_count: int = 0) -> None:
    if ready:
        st.markdown(
            f'<div class="edu-status-ok">课程助教已就绪，当前知识库包含 <b>{chunk_count}</b> 个课程片段。</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="edu-status-warn">课程知识库尚未就绪。请教师先上传资料并构建，或加载已有知识库。</div>',
            unsafe_allow_html=True,
        )


def panel_start(title: str, caption: str = "") -> None:
    st.markdown(
        f'<div class="edu-panel-title">{title}</div>'
        f'<div class="edu-muted">{caption}</div>' if caption else f'<div class="edu-panel-title">{title}</div>',
        unsafe_allow_html=True,
    )


def question_input() -> str:
    st.subheader("学生问题")
    examples = ["这门课的核心概念是什么？", "请解释本章最重要的知识点", "我应该如何复习这部分内容？"]
    for col, example in zip(st.columns(3), examples):
        if col.button(example, use_container_width=True):
            st.session_state.question_input = example
    return st.text_area(
        "输入课程相关问题",
        placeholder="例如：这门课的核心概念是什么？",
        height=130,
        key="question_input",
        label_visibility="collapsed",
    )


def render_citations(contexts: list[dict]) -> None:
    st.subheader("参考资料片段")
    for idx, item in enumerate(contexts, start=1):
        page_label = f" · 第 {item['page']} 页" if item["page"] else ""
        with st.expander(
            f"片段 {idx} · {item['source']}{page_label} · 相似度 {item['score']:.3f}",
            expanded=idx == 1,
        ):
            st.write(item["text"])
