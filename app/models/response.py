from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class BaseResponse(BaseModel):
    """基础响应模型"""
    # 添加配置以处理datetime序列化
    model_config = ConfigDict(json_encoders={
        datetime: lambda v: v.isoformat()
    })
    
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")


class ChatResponse(BaseResponse):
    """聊天响应模型"""
    content: str = Field(..., description="助手回复内容")
    conversation_id: str = Field(..., description="对话ID")
    model: str = Field(..., description="使用的模型")
    # 修复usage字段类型定义以支持OpenAI新格式
    usage: Optional[Dict[str, Any]] = Field(None, description="token使用情况")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")


class RAGResponse(BaseResponse):
    """RAG响应模型"""
    answer: str = Field(..., description="生成的答案")
    sources: List[Dict[str, Any]] = Field(..., description="参考来源")
    query: str = Field(..., description="原始查询")
    collection_name: str = Field(..., description="知识库名称")


class DocumentSource(BaseModel):
    """文档来源模型"""
    content: str = Field(..., description="文档内容")
    metadata: Dict[str, Any] = Field(..., description="文档元数据")
    score: float = Field(..., description="相似度分数")


class AgentResponse(BaseResponse):
    """Agent响应模型"""
    result: str = Field(..., description="Agent执行结果")
    agent_type: str = Field(..., description="Agent类型")
    tools_used: List[str] = Field(default_factory=list, description="使用的工具")
    execution_time: float = Field(..., description="执行时间(秒)")
    memory_used: int = Field(default=0, description="使用的记忆数量")
    trace_id: Optional[str] = Field(None, description="追踪ID")
    error: Optional[str] = Field(None, description="错误信息")


class HealthResponse(BaseModel):
    """健康检查响应模型"""
    model_config = ConfigDict(json_encoders={
        datetime: lambda v: v.isoformat()
    })
    
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="应用版本")
    timestamp: datetime = Field(default_factory=datetime.now, description="检查时间")
    services: Dict[str, str] = Field(..., description="各服务状态")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    model_config = ConfigDict(json_encoders={
        datetime: lambda v: v.isoformat()
    })
    
    error: str = Field(..., description="错误类型")
    detail: str = Field(..., description="错误详情")
    timestamp: datetime = Field(default_factory=datetime.now, description="错误时间")
    trace_id: Optional[str] = Field(None, description="追踪ID")


class FileUploadResponse(BaseResponse):
    """文件上传响应模型"""
    file_id: str = Field(..., description="文件ID")
    file_name: str = Field(..., description="文件名")
    collection_name: str = Field(..., description="知识库名称")
    chunks_count: int = Field(..., description="分块数量")
    status: str = Field(..., description="处理状态")


class MemoryResponse(BaseResponse):
    """记忆管理响应模型"""
    memories: Optional[List[Dict[str, Any]]] = Field(default=None, description="记忆列表")
    memory_count: int = Field(default=0, description="记忆数量")
    operation: str = Field(..., description="执行的操作")


class MemorySearchResponse(BaseResponse):
    """记忆搜索响应模型"""
    memories: List[Dict[str, Any]] = Field(..., description="搜索结果")
    query: str = Field(..., description="搜索查询")
    total_found: int = Field(..., description="找到的总数")


class MemorySummaryResponse(BaseResponse):
    """记忆摘要响应模型"""
    enabled: bool = Field(..., description="记忆服务是否启用")
    user_memories: Optional[int] = Field(default=None, description="用户记忆数量")
    agent_memories: Optional[int] = Field(default=None, description="Agent记忆数量")
    session_memories: Optional[int] = Field(default=None, description="会话记忆数量")
    recent_memories: Optional[List[Dict[str, Any]]] = Field(default=None, description="最近的记忆") 