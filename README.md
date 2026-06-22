# edu-rag

## 项目简介

`edu-rag` 是一个基于 Streamlit 的课程知识库问答系统，适合用于课程资料问答演示。教师上传 PDF/TXT 课程资料后，系统会解析文本、切分片段、生成向量并建立 FAISS 检索索引。学生输入问题后，系统检索最相关的课程片段，拼接为 prompt，并调用 OpenAI API 生成回答，同时显示参考资料来源。

项目还内置 SQLite 问答日志和教师端学情分析，方便查看学生提问记录、最近 7 天提问趋势和高频关键词。

## 功能说明

- 教师端
  - 上传 PDF 和 TXT 课程资料
  - 使用 PyMuPDF 解析 PDF，读取 TXT 文本
  - 使用 LangChain `RecursiveCharacterTextSplitter` 分块
  - 使用 `sentence-transformers/all-MiniLM-L6-v2` 生成文本向量
  - 使用 FAISS 构建课程知识库
  - 查看 SQLite 问答记录
  - 查看学情分析，包括总提问数、最近 7 天提问趋势、高频关键词和最近问题

- 学生端
  - 输入课程相关问题
  - 检索 Top-3 相关课程片段
  - 调用 OpenAI API 生成回答
  - 显示回答内容、参考资料来源和参考资料片段
  - 知识库为空时显示清晰提示

## 环境依赖

建议使用 Python 3.10 或更高版本。

主要依赖：

- Streamlit
- PyMuPDF
- LangChain
- sentence-transformers
- FAISS
- OpenAI Python SDK
- SQLite
- jieba
- matplotlib

完整依赖见 [requirements.txt](requirements.txt)。

## 安装步骤

```bash
cd edu-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

配置 OpenAI API Key，可任选一种方式。

方式一：设置环境变量。

```bash
export OPENAI_API_KEY="你的 OpenAI API Key"
```

方式二：启动应用后，在左侧教师端输入 API Key。

## 运行命令

```bash
streamlit run app.py
```

启动后，浏览器打开 Streamlit 提供的本地地址。

## 演示流程

1. 打开应用后，先看左侧侧边栏。
2. 在“教师端”上传 PDF 或 TXT 课程资料。
3. 填写 OpenAI API Key，确认模型名称。
4. 点击“构建知识库”，等待系统完成解析、分块、向量化和 FAISS 索引构建。
5. 点击“学生端”的“课程问答”。
6. 输入一个课程相关问题并提交。
7. 查看回答内容、参考资料来源和参考资料片段。
8. 点击“教师端”的“问答记录”，查看历史问题、回答摘要、来源和时间。
9. 点击“教师端”的“学情分析”，查看总提问数、最近 7 天趋势、高频关键词和最近问题。

## 项目结构

```text
edu-rag/
├── app.py
├── src/
│   ├── __init__.py
│   └── database.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 常见问题

### 知识库为空怎么办？

请先在左侧“教师端”上传 PDF/TXT 文件，并点击“构建知识库”。如果文件没有可解析文本，系统会提示没有解析出可用文本。

### 为什么首次运行很慢？

首次运行会下载 `sentence-transformers/all-MiniLM-L6-v2` 模型，下载完成后后续启动会更快。

### OpenAI API Key 放在哪里？

可以设置 `OPENAI_API_KEY` 环境变量，也可以在应用左侧教师端输入。没有 API Key 时，系统无法生成回答。

### 问答记录保存在哪里？

问答记录保存在项目根目录下的 `qa_logs.db`。应用启动时会自动创建数据库和 `qa_logs` 表，数据库不存在时不会报错。

### 刷新页面后知识库还在吗？

FAISS 知识库当前保存在 Streamlit 会话内存中。刷新页面或重启应用后，需要重新上传资料并构建知识库。SQLite 问答日志会保留。

### 学情分析没有数据怎么办？

学情分析读取 SQLite 中的历史提问。请先完成至少一次课程问答，生成回答后系统会自动保存日志。
