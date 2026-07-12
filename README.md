# miniKB — Mini RAG 知识库系统

> 基于 FastAPI + ChromaDB + OpenAI API 的轻量级 RAG 知识库问答系统，仿 [MaxKB](https://github.com/1Panel-dev/MaxKB) 核心架构，用于学习 RAG 原理与求职展示。

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 📄 多格式文档解析 | PDF / DOCX / TXT / Markdown / HTML |
| ✂️ 智能分块 | 递归字符分割，支持重叠（RecursiveCharacterTextSplitter） |
| 🔢 向量化存储 | OpenAI 兼容 Embedding API + ChromaDB 向量数据库 |
| 🔍 语义检索 | Cosine 相似度 Top-K 检索 |
| 💬 流式问答 | SSE 流式输出，实时显示 LLM 生成过程 |
| 📎 引用溯源 | 回答附带检索来源，可点击查看原文 |
| 🗂️ 多知识库 | 支持创建多个独立知识库，向量隔离 |
| 🤖 多模型支持 | 兼容 OpenAI / DeepSeek / Moonshot / SiliconFlow / Ollama |
| 🎨 Web UI | Vue 3 单页应用，暗色主题 |

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Vue 3)                      │
│          知识库管理 / 文档上传 / 对话界面 / 设置          │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼──────────────────────────────────┐
│                API Layer (FastAPI)                       │
│         knowledge.py / chat.py / system.py               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│               Service Layer (核心业务)                   │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────┐ │
│  │ Document │→│   Text    │→│ Embedding │→│ Vector │ │
│  │  Loader  │  │ Splitter  │  │  Service  │  │ Store  │ │
│  └──────────┘  └───────────┘  └──────────┘  └────────┘ │
│                                          ↓               │
│  ┌──────────────────────────────────────────────────┐   │
│  │                   RAG Engine                      │   │
│  │  retrieve(question) → context → LLM stream        │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Data Layer (存储)                           │
│    SQLite (metadata)  +  ChromaDB (vectors)             │
└─────────────────────────────────────────────────────────┘
```

### RAG 流程

```
用户提问
   │
   ▼
[Embedding] 将问题转为向量
   │
   ▼
[Vector Store] ChromaDB 余弦相似度检索 Top-K 文档块
   │
   ▼
[Augment] 构建系统提示词 = 检索到的上下文 + RAG 指令
   │
   ▼
[Generate] LLM 流式生成回答 (SSE → 前端实时显示)
   │
   ▼
[Cite] 附带引用来源（文档名 + 相似度分数）
```

### 设计模式

| 模式 | 应用位置 |
|------|----------|
| **分层架构** | API → Service → Data 三层解耦 |
| **Repository** | SQLAlchemy ORM 封装数据访问 |
| **Singleton** | embedding_service / vector_store / llm_provider 全局单例 |
| **Pipeline** | document_pipeline: parse → chunk → embed → store |
| **Strategy** | LLM/Embedding 通过 OpenAI 兼容接口支持多供应商 |
| **DTO** | Pydantic schemas 与 ORM models 分离 |

## 📁 项目结构

```
miniKB/
├── main.py                      # FastAPI 入口，路由注册 + 生命周期
├── config.py                    # Pydantic Settings 配置管理
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量模板
├── Dockerfile                   # Docker 镜像
├── docker-compose.yml           # Docker Compose
│
├── app/
│   ├── database.py              # SQLAlchemy 异步引擎 + Session
│   ├── models.py                # ORM 模型 (5 张表)
│   ├── schemas.py               # Pydantic 请求/响应 Schema
│   │
│   ├── routers/                 # API 路由层
│   │   ├── knowledge.py         # 知识库 CRUD + 文档上传
│   │   ├── chat.py              # 对话管理 + SSE 流式
│   │   └── system.py            # 健康检查 + 配置 + 统计
│   │
│   ├── services/                # 核心服务层
│   │   ├── document_loader.py   # 文档解析 (PDF/DOCX/TXT/MD/HTML)
│   │   ├── text_splitter.py     # 递归字符分块
│   │   ├── embedding.py         # OpenAI 兼容 Embedding
│   │   ├── vector_store.py      # ChromaDB 向量存储
│   │   ├── llm_provider.py      # LLM 流式生成
│   │   ├── rag_engine.py        # RAG 管道编排
│   │   └── document_pipeline.py # 文档处理流水线
│   │
│   └── static/                  # 前端静态资源
│       ├── index.html           # Vue 3 SPA
│       ├── style.css            # 暗色主题样式
│       └── app.js               # 前端逻辑
│
└── data/                        # 运行时数据 (gitignored)
    ├── uploads/                 # 上传的文件
    ├── chroma/                  # ChromaDB 向量数据
    └── miniKB.db                # SQLite 元数据
```

## 🚀 快速开始

### 方式一：本地运行

```bash
# 1. 克隆项目
cd D:\pycdoe\miniKB

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key（推荐 DeepSeek，便宜好用）

# 5. 启动
python main.py
# 或: uvicorn main:app --reload --port 8000

# 6. 打开浏览器
# http://localhost:8000
```

### 方式二：Docker 运行

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 API Key

# 2. 一键启动
docker compose up -d

# 3. 打开浏览器
# http://localhost:8000
```

## 🔑 API Key 配置

miniKB 兼容所有 OpenAI 格式的 API，推荐使用 [DeepSeek](https://platform.deepseek.com/)：

```env
# DeepSeek（推荐，便宜）
LLM_API_KEY=sk-your-deepseek-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
EMBEDDING_API_KEY=sk-your-deepseek-key
EMBEDDING_BASE_URL=https://api.deepseek.com/v1
EMBEDDING_MODEL=text-embedding-3-small
```

其他兼容服务：

| 服务商 | BASE_URL | 模型示例 |
|--------|----------|----------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Moonshot | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| SiliconFlow | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-7B-Instruct` |
| Ollama (本地) | `http://localhost:11434/v1` | `llama3.2` |

## 📡 API 文档

启动后访问 `http://localhost:8000/docs` 查看自动生成的 Swagger 文档。

主要接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/kb` | 创建知识库 |
| GET | `/api/kb` | 列出知识库 |
| POST | `/api/kb/{id}/upload` | 上传文档 |
| GET | `/api/kb/{id}/docs` | 列出文档 |
| POST | `/api/chat` | 创建对话 |
| POST | `/api/chat/{id}/send` | 发送消息（SSE 流式） |
| GET | `/api/system/stats` | 仪表盘统计 |

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI 0.115 + Uvicorn |
| 数据库 | SQLite + SQLAlchemy 2.0 (async) |
| 向量数据库 | ChromaDB |
| LLM / Embedding | OpenAI Python SDK (兼容多供应商) |
| 文档解析 | pypdf / python-docx / beautifulsoup4 |
| 分块策略 | LangChain RecursiveCharacterTextSplitter |
| 前端 | Vue 3 (CDN) + 原生 CSS |
| 部署 | Docker + Docker Compose |

## 📝 使用流程

1. **配置模型** — 编辑 `.env` 填入 API Key
2. **创建知识库** — 点击侧栏 + 按钮
3. **上传文档** — 拖拽或点击上传 PDF/Word/TXT
4. **等待处理** — 系统自动分块 + 向量化
5. **开始对话** — 新建对话，输入问题，获得引用回答

## 📄 License

MIT
