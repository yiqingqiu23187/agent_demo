# AI Agent 应用开发演示项目

这是一个基于FastAPI的AI Agent应用开发演示项目，集成了多种主流技术栈，用于展示AI Agent应用的完整开发流程。

## 🎯 项目特色

### 集成的技术栈
- **🤖 LLM模型**: OpenAI GPT-4o、Anthropic Claude-3.5
- **🔗 Agent框架**: LangChain ReAct、CrewAI多智能体、Dify低代码平台
- **📊 监控系统**: LangFuse 调用链追踪和性能监控
- **🗄️ 向量数据库**: Chroma、Qdrant、Pinecone
- **🛠️ 工具集成**: 搜索引擎、计算器、API调用、文件处理
- **🚀 Web框架**: FastAPI + Uvicorn

### 核心功能
- **🤖 多Agent架构**: LangChain ReAct Agent、Dify Agent、CrewAI多智能体协作
- **🔍 知识库管理**: 文档上传、向量存储、RAG检索
- **🛠️ 工具调用**: 搜索、计算、API调用等多种工具集成
- **📈 性能监控**: LangFuse集成的完整可观测性
- **🧠 记忆系统**: 用户对话历史和上下文记忆
- **🐳 容器化部署**: Docker Compose一键启动

## 📁 项目结构

```
agent_demo/
├── AI_Agent_Development_Guide.md  # 技术分享文档
├── app/                           # FastAPI应用
│   ├── main.py                    # 应用入口
│   ├── core/                      # 核心服务 (LLM、RAG、内存、监控等)
│   ├── agents/                    # 智能体实现 (LangChain、Dify、CrewAI)
│   ├── api/v1/                    # API路由 (专注Agent和知识库管理)
│   ├── models/                    # 数据模型
│   └── utils/                     # 工具函数
├── requirements.txt               # Python依赖
├── docker-compose.yml             # 容器编排
├── .env.example                   # 环境变量示例
└── README.md                      # 项目说明
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd agent_demo

# 复制环境变量文件
cp .env.example .env
```

### 2. 配置环境变量

编辑 `.env` 文件，配置必要的API密钥：

```bash
# LLM API Keys
OPENAI_API_KEY=sk-your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# LangFuse监控 (可选)
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key

# 其他配置保持默认即可
```

### 3. 启动方式

#### 方式一：本地开发
```bash
# 安装依赖
pip install -r requirements.txt

# 启动应用
python -m app.main
```

#### 方式二：Docker Compose (推荐)
```bash
# 一键启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f app
```

### 4. 访问服务

- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **Dify界面**: http://localhost:3000 (可选)
- **Qdrant控制台**: http://localhost:6333/dashboard

## 📡 API 使用示例

### 知识库管理
```bash
# 添加文档到知识库
curl -X POST "http://localhost:8000/api/v1/rag/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "demo_kb",
    "texts": ["AI Agent是具备记忆、工具调用、推理决策能力的智能系统"],
    "metadatas": [{"source": "demo"}]
  }'

# 批量添加文档
curl -X POST "http://localhost:8000/api/v1/rag/batch_documents" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "demo_kb",
    "documents": [
      {"text": "文档内容1", "metadata": {"type": "article"}},
      {"text": "文档内容2", "metadata": {"type": "guide"}}
    ]
  }'
```

### Agent执行
```bash
# LangChain ReAct Agent
curl -X POST "http://localhost:8000/api/v1/agents/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "帮我制定一个学习AI的计划",
    "agent_type": "react",
    "tools": ["search", "calculator"],
    "use_memory": true
  }'

# CrewAI 多智能体协作
curl -X POST "http://localhost:8000/api/v1/agents/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "分析当前AI技术趋势并撰写报告",
    "agent_type": "crew",
    "metadata": {
      "agents": ["researcher", "analyst", "writer"],
      "process": "sequential"
    }
  }'

# Dify Agent
curl -X POST "http://localhost:8000/api/v1/agents/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "处理客户咨询",
    "agent_type": "dify",
    "metadata": {
      "workflow_id": "your-workflow-id"
    }
  }'
```

## 🏗️ 技术架构

### 服务层架构
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   FastAPI   │───▶│   LLM服务   │───▶│ OpenAI/Claude│
│   API层     │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  RAG服务    │───▶│  向量服务   │───▶│Chroma/Qdrant│
│             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
       │
       ▼
┌─────────────┐    ┌─────────────┐
│  监控服务   │───▶│  LangFuse   │
│             │    │             │
└─────────────┘    └─────────────┘
```

### 主要特性

#### 🔧 模块化设计
- **配置管理**: 基于Pydantic的类型安全配置
- **服务分离**: LLM、向量数据库、监控等独立服务
- **统一接口**: 标准化的API响应格式

#### 📊 可观测性
- **调用追踪**: LangFuse集成的完整调用链
- **性能监控**: 响应时间、token使用量统计
- **错误监控**: 异常捕获和错误追踪

#### 🔒 生产就绪
- **异常处理**: 全局异常捕获和标准化错误响应
- **健康检查**: 服务状态监控端点
- **日志系统**: 结构化日志记录
- **环境隔离**: 开发/生产环境配置分离

## 🛠️ 开发指南

### 添加新的LLM模型
1. 在 `app/services/llm_service.py` 中添加模型初始化
2. 更新 `app/core/config.py` 中的配置项
3. 测试模型可用性

### 添加新的向量数据库
1. 继承 `VectorStoreBase` 类
2. 实现必需的抽象方法
3. 在 `VectorService` 中注册新的存储类型

### 扩展Agent功能
1. 在 `app/api/v1/agents.py` 中添加新的Agent类型
2. 实现对应的执行逻辑
3. 更新API文档和测试用例

## 🔍 监控和调试

### LangFuse监控面板
访问 LangFuse 控制台查看：
- 调用链追踪 (Traces)
- 性能指标 (Metrics)  
- 成本分析 (Cost Analysis)
- 模型评估 (Evaluations)

### 日志查看
```bash
# Docker环境
docker-compose logs -f app

# 本地环境
tail -f logs/app.log
```

### 健康检查
```bash
curl http://localhost:8000/health
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙋‍♂️ 支持

如有问题或建议，请：
- 创建 [Issue](../../issues)
- 发送邮件至项目维护者
- 查看技术文档：[AI_Agent_Development_Guide.md](./AI_Agent_Development_Guide.md)

---

**🎉 开始你的AI Agent开发之旅吧！** 