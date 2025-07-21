from typing import Optional
import time
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.request import AgentRequest, DifyRequest
from app.models.response import AgentResponse
from app.services.agent_service import enhanced_agent_service
from app.services.monitoring_service import monitoring_service
from app.utils.logger import app_logger
import httpx

router = APIRouter()


@router.post("/agents/execute", response_model=AgentResponse)
async def execute_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks
):
    """执行Agent任务"""
    try:
        if request.agent_type == "react":
            # 执行ReAct Agent
            result = await enhanced_agent_service.execute_react_agent(
                task=request.task,
                user_id=request.user_id,
                session_id=request.session_id,
                agent_id=request.agent_id,
                model=request.model,
                max_iterations=request.max_iterations,
                use_memory=request.use_memory
            )
        elif request.agent_type == "simple":
            # 执行简单Agent
            result = await enhanced_agent_service.execute_simple_agent(
                task=request.task,
                user_id=request.user_id,
                session_id=request.session_id,
                agent_id=request.agent_id,
                model=request.model,
                use_memory=request.use_memory
            )
        elif request.agent_type == "dify":
            # 保持原有的Dify实现
            result = await _execute_dify_agent(request)
        else:
            raise ValueError(f"不支持的Agent类型: {request.agent_type}")
        
        # 后台任务：刷新监控数据
        background_tasks.add_task(monitoring_service.flush)
        
        return AgentResponse(
            success=result["success"],
            message="Agent执行完成",
            result=result["result"],
            agent_type=result["agent_type"],
            steps=result.get("steps", []),
            tools_used=result.get("tools_used", []),
            execution_time=result["execution_time"],
            memory_used=result.get("memory_used", 0),
            trace_id=result.get("trace_id")
        )
        
    except Exception as e:
        app_logger.error(f"Agent执行失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _execute_dify_agent(request: AgentRequest) -> dict:
    """执行Dify Agent"""
    try:
        # 这里是Dify API调用的简化实现
        # 实际项目中应该调用真正的Dify API
        
        steps = [
            {"step": 1, "action": "连接Dify", "result": "已连接到Dify平台"},
            {"step": 2, "action": "发送任务", "result": f"任务已发送: {request.task}"},
            {"step": 3, "action": "处理响应", "result": "Dify正在处理..."},
            {"step": 4, "action": "返回结果", "result": "获得Dify响应"}
        ]
        
        tools_used = ["dify_workflow"]
        
        result = f"基于Dify平台的执行结果：已完成任务 '{request.task}'"
        
        return {
            "success": True,
            "result": result,
            "agent_type": "dify",
            "steps": steps,
            "tools_used": tools_used,
            "execution_time": 1.0,
            "memory_used": 0
        }
        
    except Exception as e:
        app_logger.error(f"Dify Agent执行失败: {e}")
        raise


@router.post("/agents/dify/chat")
async def dify_chat(request: DifyRequest):
    """Dify聊天接口"""
    try:
        # 这里应该调用实际的Dify API
        # 由于演示项目，返回模拟响应
        
        response = {
            "success": True,
            "answer": f"Dify回复: {request.query}",
            "conversation_id": request.conversation_id or "dify-conv-123",
            "message_id": "msg-123",
            "metadata": {
                "usage": {"tokens": 100},
                "model": "dify-model"
            }
        }
        
        return response
        
    except Exception as e:
        app_logger.error(f"Dify聊天失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/types")
async def get_agent_types():
    """获取支持的Agent类型"""
    return {
        "success": True,
        "agent_types": [
            {
                "type": "react",
                "name": "ReAct Agent",
                "description": "基于ReAct模式的智能体，支持推理和行动循环",
                "capabilities": ["工具调用", "推理链", "记忆管理", "多步骤执行"]
            },
            {
                "type": "simple", 
                "name": "Simple Agent",
                "description": "简单的对话式智能体，基于记忆增强",
                "capabilities": ["对话生成", "记忆管理", "上下文理解"]
            },
            {
                "type": "dify",
                "name": "Dify Agent", 
                "description": "基于Dify平台的智能体",
                "capabilities": ["可视化工作流", "多模型支持", "低代码开发"]
            }
        ]
    }


@router.get("/agents/tools")
async def get_available_tools():
    """获取可用工具列表"""
    try:
        tools = enhanced_agent_service.get_available_tools()
        return {
            "success": True,
            "tools": tools,
            "count": len(tools)
        }
    except Exception as e:
        app_logger.error(f"获取工具列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/memory/summary")
async def get_agent_memory_summary(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """获取Agent记忆摘要"""
    try:
        summary = await enhanced_agent_service.get_memory_summary(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id
        )
        return {
            "success": True,
            "memory_summary": summary
        }
    except Exception as e:
        app_logger.error(f"获取记忆摘要失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 