from typing import Optional
import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from app.models.request import ChatRequest
from app.models.response import ChatResponse, ErrorResponse
from app.services.llm_service import llm_service
from app.services.monitoring_service import monitoring_service
from app.utils.logger import app_logger
from langchain_core.messages import HumanMessage, SystemMessage

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = None
):
    """聊天补全接口"""
    try:
        # 生成对话ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # 获取LangFuse回调处理器
        langfuse_handler = monitoring_service.get_langchain_callback_handler()
        callbacks = [langfuse_handler] if langfuse_handler else []
        
        # 检查模型可用性
        if request.model not in llm_service._models:
            raise ValueError(f"模型 {request.model} 不可用")
        
        llm = llm_service._models[request.model]
        
        # 构建消息
        messages = [HumanMessage(content=request.message)]
        
        # 调用LLM（通过LangChain，自动追踪到LangFuse）
        response = await llm.ainvoke(
            messages,
            config={
                "callbacks": callbacks,
                "metadata": {
                    "langfuse_user_id": user_id,
                    "langfuse_session_id": conversation_id,
                    "langfuse_tags": ["chat-completion"],
                    "model": request.model,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                }
            }
        )
        
        # 后台任务：刷新监控数据
        background_tasks.add_task(monitoring_service.flush)
        
        return ChatResponse(
            success=True,
            message="聊天完成",
            content=response.content,
            conversation_id=conversation_id,
            model=request.model,
            usage=getattr(response, 'usage_metadata', {}),
            metadata={
                "finish_reason": getattr(response, 'response_metadata', {}).get('finish_reason'),
                "trace_id": langfuse_handler.last_trace_id if langfuse_handler else None
            }
        )
        
    except Exception as e:
        app_logger.error(f"聊天补全失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def stream_chat_completion(
    request: ChatRequest,
    user_id: Optional[str] = None
):
    """流式聊天补全接口"""
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        async def generate_stream():
            try:
                # 获取LangFuse回调处理器
                langfuse_handler = monitoring_service.get_langchain_callback_handler()
                callbacks = [langfuse_handler] if langfuse_handler else []
                
                # 检查模型可用性
                if request.model not in llm_service._models:
                    raise ValueError(f"模型 {request.model} 不可用")
                
                llm = llm_service._models[request.model]
                
                # 构建消息
                messages = [HumanMessage(content=request.message)]
                
                # 流式调用LLM（通过LangChain，自动追踪到LangFuse）
                async for chunk in llm.astream(
                    messages,
                    config={
                        "callbacks": callbacks,
                        "metadata": {
                            "langfuse_user_id": user_id,
                            "langfuse_session_id": conversation_id,
                            "langfuse_tags": ["chat-stream"],
                            "model": request.model,
                            "temperature": request.temperature,
                            "streaming": True
                        }
                    }
                ):
                    if chunk.content:
                        yield f"data: {chunk.content}\n\n"
                yield "data: [DONE]\n\n"
                
                # 刷新监控数据
                monitoring_service.flush()
                
            except Exception as e:
                app_logger.error(f"流式聊天失败: {e}")
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Conversation-ID": conversation_id
            }
        )
        
    except Exception as e:
        app_logger.error(f"流式聊天初始化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_available_models():
    """获取可用模型列表"""
    try:
        models = llm_service.get_available_models()
        return {
            "success": True,
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        app_logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """获取模型信息"""
    try:
        model_info = llm_service.get_model_info(model_name)
        
        if "error" in model_info:
            raise HTTPException(status_code=404, detail=model_info["error"])
        
        return {
            "success": True,
            "model_info": model_info
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 