from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str = Field(..., description="消息角色: user, assistant, system")
    content: str = Field(..., description="消息内容")
    
    
class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户消息")
    conversation_id: Optional[str] = Field(None, description="对话ID")
    model: str = Field(default="gpt-4o-mini", description="使用的模型")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: Optional[int] = Field(default=None, description="最大token数")
    stream: bool = Field(default=False, description="是否流式输出")


class RAGQueryRequest(BaseModel):
    """RAG查询请求模型"""
    query: str = Field(..., description="查询内容")
    collection_name: str = Field(..., description="知识库名称")
    top_k: int = Field(default=5, ge=1, le=20, description="检索数量")
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="相似度阈值")


class DocumentUploadRequest(BaseModel):
    """文档上传请求模型"""
    file_name: str = Field(..., description="文件名")
    collection_name: str = Field(..., description="知识库名称")
    chunk_size: int = Field(default=1000, description="文档分块大小")
    chunk_overlap: int = Field(default=200, description="分块重叠大小")


class AgentRequest(BaseModel):
    """Agent请求模型"""
    task: str = Field(..., description="任务描述")
    agent_type: str = Field(default="react", description="Agent类型: react, crew, dify")
    model: str = Field(default="gpt-4o-mini", description="使用的模型")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    agent_id: Optional[str] = Field(default=None, description="Agent ID")
    tools: Optional[List[str]] = Field(default=None, description="可用工具列表")
    use_memory: bool = Field(default=True, description="是否启用记忆")
    max_iterations: int = Field(default=5, ge=1, le=20, description="最大迭代次数")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="额外元数据")


class DifyRequest(BaseModel):
    """Dify API请求模型"""
    query: str = Field(..., description="查询内容")
    user: str = Field(..., description="用户标识")
    response_mode: str = Field(default="blocking", description="响应模式: blocking, streaming")
    conversation_id: Optional[str] = Field(None, description="对话ID")
    inputs: Optional[Dict[str, Any]] = Field(default=None, description="输入变量")
    files: Optional[List[Dict[str, str]]] = Field(default=None, description="文件列表")


class MemoryRequest(BaseModel):
    """记忆管理请求模型"""
    messages: List[Dict[str, str]] = Field(..., description="消息列表")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    agent_id: Optional[str] = Field(default=None, description="Agent ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="元数据")


class MemorySearchRequest(BaseModel):
    """记忆搜索请求模型"""
    query: str = Field(..., description="搜索查询")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    agent_id: Optional[str] = Field(default=None, description="Agent ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    limit: int = Field(default=5, ge=1, le=20, description="返回结果数量")


class MemoryDeleteRequest(BaseModel):
    """记忆删除请求模型"""
    memory_id: Optional[str] = Field(default=None, description="指定记忆ID")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    agent_id: Optional[str] = Field(default=None, description="Agent ID")
    session_id: Optional[str] = Field(default=None, description="会话ID")
    delete_all: bool = Field(default=False, description="是否删除所有记忆") 