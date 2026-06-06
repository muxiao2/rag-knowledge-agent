# RAG 知识库问答 Agent

[English](./README.md) | 简体中文

一个**完全本地、离线优先**的 **RAG（检索增强生成）知识库问答 Agent**。把你的文档放进一个文件夹，导入本地向量数据库后即可提问——回答会基于你自己的知识库内容，并**标注出处**。全部通过 [Ollama](https://ollama.com) 在本机运行，**数据不出本地**。

## 功能特性

- 🧠 **本地大模型 + 向量化**：基于 Ollama（默认 `llama3.1` + `nomic-embed-text`）
- 📚 **文档导入**：支持 `.md`、`.txt`、`.pdf`
- 🔎 **本地向量库**：使用 [ChromaDB](https://www.trychroma.com/)，持久化到磁盘
- 💬 **命令行（CLI）**：`ask`、`chat`、`ingest`、`serve`、`info`（基于 Typer + Rich）
- 🌐 **HTTP 接口**：基于 FastAPI（`/ask`、`/ingest`、`/health`）
- ✅ 配套单元测试、代码风格检查（ruff）与 GitHub Actions CI

## 工作原理（架构）

```
            ┌──────────────┐      ┌───────────────┐      ┌──────────────┐
 文档      → │   loaders    │ ───► │   chunking    │ ───► │  embeddings  │
(.md/.txt/  │  (读取文件)  │      │  (文本分块)   │      │   (Ollama)   │
 .pdf)      └──────────────┘      └───────────────┘      └──────┬───────┘
                                                                │
                                                          ┌─────▼──────┐
                                                          │  ChromaDB  │  (持久化)
                                                          └─────┬──────┘
   问题   ────────────────────────────────────────────────────►│ 检索 top-k
                                                          ┌─────▼──────┐
                                                          │  ChatOllama│ → 基于上下文的答案
                                                          │ (llama3.1) │    + 出处
                                                          └────────────┘
```

整体流程为 `loaders → chunking → embeddings → Chroma → 检索 → ChatOllama`：

1. **导入（Ingest）**：读取 `data/docs/` 下的文档，切分为约 800 字符、相互重叠的文本块。
2. **向量化（Embed）**：用 Ollama 向量模型 `nomic-embed-text` 把每个文本块转成向量。
3. **存储（Store）**：向量持久化保存到本地 ChromaDB 集合中。
4. **检索（Retrieve）**：提问时取回最相关的若干文本块。
5. **生成（Generate）**：把文本块作为上下文交给 Ollama 对话模型 `llama3.1`，生成基于上下文、可溯源的答案。

## 环境要求

1. **Python 3.11+** 以及 [`uv`](https://docs.astral.sh/uv/)（推荐）或 `pip`。
2. 安装并运行 **Ollama**：https://ollama.com/download
3. 拉取默认模型：

   ```bash
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```

## 安装

```bash
# 克隆仓库后：
uv sync --extra dev          # 安装依赖（含开发工具）
cp .env.example .env         # 可选：自定义模型 / 参数
```

> 想用 pip？`python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`

## 使用

### 1. 放入你的文档

把 `.md`、`.txt`、`.pdf` 文件放到 `data/docs/` 目录（已自带一个示例文件）。

### 2. 导入到向量库

```bash
uv run rag-agent ingest --reset
```

### 3. 提问

```bash
uv run rag-agent ask "RAG 是什么？这个项目是怎么工作的？"
```

### 4. 交互式对话

```bash
uv run rag-agent chat
```

### 5. 启动 HTTP 服务

```bash
uv run rag-agent serve          # http://localhost:8000（接口文档在 /docs）
```

```bash
# 先导入，再通过 HTTP 提问
curl -X POST localhost:8000/ingest
curl -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "这个项目默认用什么模型？"}'
```

### 查看当前配置

```bash
uv run rag-agent info
```

## 配置项

所有配置均为带 `RAG_` 前缀的环境变量（见 `.env.example`）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RAG_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `RAG_LLM_MODEL` | `llama3.1` | 对话模型 |
| `RAG_EMBEDDING_MODEL` | `nomic-embed-text` | 向量化模型 |
| `RAG_CHUNK_SIZE` | `800` | 每个文本块的字符数 |
| `RAG_CHUNK_OVERLAP` | `120` | 文本块之间的重叠字符数 |
| `RAG_TOP_K` | `4` | 每次检索返回的文本块数量 |
| `RAG_COLLECTION_NAME` | `knowledge_base` | Chroma 集合名称 |
| `RAG_API_HOST` / `RAG_API_PORT` | `0.0.0.0` / `8000` | API 监听地址 |

## 目录结构

```
src/rag_agent/
  config.py       # 配置（pydantic-settings）
  loaders.py      # 读取 .md/.txt/.pdf 为 Document
  chunking.py     # 零依赖的文本分块器
  vectorstore.py  # Ollama 向量化 + Chroma
  ingest.py       # 导入流水线：读取 → 分块 → 向量化 → 持久化
  rag.py          # 检索 + 生成（核心 Agent）
  prompts.py      # 提示词模板
  api.py          # FastAPI 应用
  cli.py          # Typer 命令行
tests/            # 单元测试（无需 Ollama 即可运行）
data/docs/        # 你的知识库文档放在这里
```

## 开发

```bash
make dev     # 安装开发依赖
make lint    # ruff 检查
make test    # 运行 pytest
make fmt     # ruff 格式化
```

## 常见问题

**这个项目需要联网吗？**
不需要。模型拉取完成后，一切都在本地离线运行。

**支持哪些文件类型？**
纯文本（`.txt`）、Markdown（`.md`/`.markdown`）、PDF（`.pdf`）。

**怎么更换模型？**
在 `.env` 文件中设置 `RAG_LLM_MODEL` 和 `RAG_EMBEDDING_MODEL`。

**回答不准 / 找不到内容怎么办？**
适当调大 `RAG_TOP_K`，或调整 `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` 后重新执行 `rag-agent ingest --reset`。

## 许可证

MIT
