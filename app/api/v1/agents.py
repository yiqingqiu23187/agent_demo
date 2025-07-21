from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.request import AgentRequest, DifyRequest
from app.models.response import AgentResponse
from app.agents.langchain_react_agent import langchain_react_agent
from app.agents.dify_agent import dify_agent
from app.agents.crew_multi_agent import crew_multi_agent
from app.core.monitoring_service import monitoring_service
from app.utils.logger import app_logger

router = APIRouter()


@router.post("/agents/execute", response_model=AgentResponse)
async def execute_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks
):
    """执行Agent任务"""
    try:
        # 根据agent类型调用对应的agent
        if request.agent_type == "react":
            result = await langchain_react_agent.execute(
                task=request.task,
                user_id=request.user_id,
                session_id=request.session_id,
                agent_id=request.agent_id,
                model=request.model,
                max_iterations=request.max_iterations,
                use_memory=request.use_memory,
                selected_tools=request.tools
            )
        elif request.agent_type == "crew":
            # Crew多智能体协作
            crew_config = None
            if request.metadata:
                crew_config = {
                    "agents": request.metadata.get("agents", ["researcher", "analyst", "writer"]),
                    "process": request.metadata.get("process", "sequential"),
                    "max_iterations": request.metadata.get("max_iterations", 5)
                }
            
            result = await crew_multi_agent.execute(
                task=request.task,
                user_id=request.user_id,
                session_id=request.session_id,
                agent_id=request.agent_id,
                model=request.model,
                use_memory=request.use_memory,
                selected_tools=request.tools,
                crew_config=crew_config
            )
        elif request.agent_type == "dify":
            # 从metadata中提取workflow_id
            workflow_id = None
            if request.metadata:
                workflow_id = request.metadata.get("workflow_id")
            
            # 构建dify参数
            dify_kwargs = {}
            if request.metadata:
                dify_kwargs.update(request.metadata)
            
            result = await dify_agent.execute(
                task=request.task,
                user_id=request.user_id,
                session_id=request.session_id,
                agent_id=request.agent_id,
                workflow_id=workflow_id,
                use_memory=request.use_memory,
                **dify_kwargs
            )
        else:
            raise ValueError(f"不支持的Agent类型: {request.agent_type}。支持的类型: react, crew, dify")
        
        # 后台任务：刷新监控数据
        background_tasks.add_task(monitoring_service.flush)
        
        return AgentResponse(
            success=result["success"],
            message="Agent执行完成" if result["success"] else "Agent执行失败",
            result=result["result"],
            agent_type=result["agent_type"],
            tools_used=result.get("tools_used", []),
            execution_time=result["execution_time"],
            memory_used=result.get("memory_used", 0),
            trace_id=result.get("trace_id"),
            error=result.get("error")
        )
        
    except Exception as e:
        app_logger.error(f"Agent执行失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/types")
async def get_available_agent_types():
    """获取所有可用的Agent类型"""
    try:
        agents_info = [
            langchain_react_agent.get_agent_info(),
            crew_multi_agent.get_agent_info(),
            dify_agent.get_agent_info()
        ]
        
        return {
            "success": True,
            "message": "获取Agent类型成功",
            "agents": agents_info,
            "supported_types": ["react", "crew", "dify"]
        }
    except Exception as e:
        app_logger.error(f"获取Agent类型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/types/{agent_type}")
async def get_agent_info(agent_type: str):
    """获取特定Agent类型的详细信息"""
    try:
        if agent_type == "react":
            agent_info = langchain_react_agent.get_agent_info()
        elif agent_type == "crew":
            agent_info = crew_multi_agent.get_agent_info()
        elif agent_type == "dify":
            agent_info = dify_agent.get_agent_info()
        else:
            raise HTTPException(status_code=404, detail=f"不支持的Agent类型: {agent_type}。支持的类型: react, crew, dify")
        
        return {
            "success": True,
            "agent_info": agent_info
        }
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"获取Agent信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/crew/roles")
async def get_crew_roles():
    """获取Crew多智能体的预定义角色"""
    try:
        roles = crew_multi_agent.get_predefined_roles()
        return {
            "success": True,
            "message": "获取Crew角色成功",
            "roles": roles
        }
    except Exception as e:
        app_logger.error(f"获取Crew角色失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/dify/workflows")
async def list_dify_workflows():
    """获取Dify工作流列表"""
    try:
        workflows = await dify_agent.list_workflows()
        return {
            "success": True,
            "message": "获取Dify工作流列表成功",
            "workflows": workflows
        }
    except Exception as e:
        app_logger.error(f"获取Dify工作流列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/dify/chat")
async def dify_chat_completion(
    request: DifyRequest,
    background_tasks: BackgroundTasks
):
    """Dify聊天补全接口（兼容性接口）"""
    try:
        # 直接调用dify_agent
        result = await dify_agent.execute(
            task=request.query,
            user_id=request.user,
            workflow_id=request.workflow_id,
            **request.inputs
        )
        
        # 后台任务：刷新监控数据
        background_tasks.add_task(monitoring_service.flush)
        
        return {
            "success": result["success"],
            "message": result["result"],
            "workflow_id": request.workflow_id,
            "execution_time": result["execution_time"]
        }
        
    except Exception as e:
        app_logger.error(f"Dify聊天失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/health")
async def agents_health_check():
    """Agent服务健康检查"""
    try:
        from app.core.memory_service import memory_service
        
        health_info = {
            "supported_agents": ["react", "crew", "dify"],
            "agents_count": 3,
            "memory_service": "enabled" if memory_service.is_enabled() else "disabled",
            "monitoring_service": "enabled" if monitoring_service.is_enabled() else "disabled",
            "langchain_react_agent": "healthy",
            "crew_multi_agent": "healthy",
            "dify_agent": "healthy"
        }
        
        return {
            "success": True,
            "message": "Agent服务运行正常",
            "health_info": health_info
        }
    except Exception as e:
        app_logger.error(f"Agent健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 