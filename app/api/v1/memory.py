from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.request import MemoryRequest, MemorySearchRequest, MemoryDeleteRequest
from app.models.response import MemoryResponse, MemorySearchResponse, MemorySummaryResponse
from app.services.memory_service import memory_service
from app.utils.logger import app_logger

router = APIRouter()


@router.post("/memory/add", response_model=MemoryResponse)
async def add_memory(
    request: MemoryRequest,
    background_tasks: BackgroundTasks
):
    """添加记忆"""
    try:
        if not memory_service.is_enabled():
            raise HTTPException(status_code=503, detail="记忆服务未启用")
        
        # 至少需要一个ID
        if not any([request.user_id, request.agent_id, request.session_id]):
            raise HTTPException(status_code=400, detail="必须提供user_id、agent_id或session_id中的至少一个")
        
        results = []
        
        # 添加用户记忆
        if request.user_id:
            result = await memory_service.add_user_memory(
                messages=request.messages,
                user_id=request.user_id,
                metadata=request.metadata
            )
            if result["success"]:
                results.extend(result.get("memories", []))
        
        # 添加Agent记忆
        if request.agent_id:
            result = await memory_service.add_agent_memory(
                messages=request.messages,
                agent_id=request.agent_id,
                metadata=request.metadata
            )
            if result["success"]:
                results.extend(result.get("memories", []))
        
        # 添加会话记忆
        if request.session_id:
            result = await memory_service.add_session_memory(
                messages=request.messages,
                session_id=request.session_id,
                metadata=request.metadata
            )
            if result["success"]:
                results.extend(result.get("memories", []))
        
        return MemoryResponse(
            success=True,
            message="记忆添加成功",
            memories=results,
            memory_count=len(results),
            operation="add"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"添加记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(request: MemorySearchRequest):
    """搜索记忆"""
    try:
        if not memory_service.is_enabled():
            raise HTTPException(status_code=503, detail="记忆服务未启用")
        
        all_memories = []
        
        # 搜索用户记忆
        if request.user_id:
            user_memories = await memory_service.search_user_memories(
                query=request.query,
                user_id=request.user_id,
                limit=request.limit
            )
            all_memories.extend(user_memories)
        
        # 搜索Agent记忆
        if request.agent_id:
            agent_memories = await memory_service.search_agent_memories(
                query=request.query,
                agent_id=request.agent_id,
                limit=request.limit
            )
            all_memories.extend(agent_memories)
        
        # 搜索会话记忆
        if request.session_id:
            session_memories = await memory_service.search_session_memories(
                query=request.query,
                session_id=request.session_id,
                limit=request.limit
            )
            all_memories.extend(session_memories)
        
        # 如果没有指定任何ID，返回错误
        if not any([request.user_id, request.agent_id, request.session_id]):
            raise HTTPException(status_code=400, detail="必须提供user_id、agent_id或session_id中的至少一个")
        
        # 去重并按相关性排序
        unique_memories = []
        seen_ids = set()
        for memory in all_memories:
            memory_id = memory.get("id")
            if memory_id and memory_id not in seen_ids:
                unique_memories.append(memory)
                seen_ids.add(memory_id)
        
        # 限制返回数量
        unique_memories = unique_memories[:request.limit]
        
        return MemorySearchResponse(
            success=True,
            message="记忆搜索完成",
            memories=unique_memories,
            query=request.query,
            total_found=len(unique_memories)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"搜索记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/list")
async def list_memories(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 10
):
    """获取记忆列表"""
    try:
        if not memory_service.is_enabled():
            raise HTTPException(status_code=503, detail="记忆服务未启用")
        
        if not any([user_id, agent_id, session_id]):
            raise HTTPException(status_code=400, detail="必须提供user_id、agent_id或session_id中的至少一个")
        
        all_memories = []
        
        # 获取用户记忆
        if user_id:
            user_memories = await memory_service.get_all_user_memories(
                user_id=user_id,
                limit=limit
            )
            all_memories.extend(user_memories)
        
        # 获取Agent记忆
        if agent_id:
            agent_memories = await memory_service.get_all_agent_memories(
                agent_id=agent_id,
                limit=limit
            )
            all_memories.extend(agent_memories)
        
        return {
            "success": True,
            "memories": all_memories[:limit],
            "total_count": len(all_memories),
            "user_id": user_id,
            "agent_id": agent_id,
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"获取记忆列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memory/delete")
async def delete_memory(request: MemoryDeleteRequest):
    """删除记忆"""
    try:
        if not memory_service.is_enabled():
            raise HTTPException(status_code=503, detail="记忆服务未启用")
        
        if request.delete_all:
            # 删除所有记忆
            if request.user_id:
                result = await memory_service.delete_all_user_memories(request.user_id)
                if not result["success"]:
                    raise HTTPException(status_code=500, detail=result["message"])
            
            if request.session_id:
                result = await memory_service.delete_all_session_memories(request.session_id)
                if not result["success"]:
                    raise HTTPException(status_code=500, detail=result["message"])
            
            return {
                "success": True,
                "message": "记忆删除成功",
                "operation": "delete_all"
            }
        
        elif request.memory_id:
            # 删除指定记忆
            result = await memory_service.delete_memory(request.memory_id)
            if not result["success"]:
                raise HTTPException(status_code=500, detail=result["message"])
            
            return {
                "success": True,
                "message": "记忆删除成功",
                "operation": "delete_single",
                "memory_id": request.memory_id
            }
        
        else:
            raise HTTPException(status_code=400, detail="必须提供memory_id或设置delete_all=True")
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"删除记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/reset")
async def reset_all_memories():
    """重置所有记忆"""
    try:
        if not memory_service.is_enabled():
            raise HTTPException(status_code=503, detail="记忆服务未启用")
        
        result = memory_service.reset_all_memories()
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        
        return {
            "success": True,
            "message": "所有记忆已重置",
            "operation": "reset_all"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"重置记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/summary", response_model=MemorySummaryResponse)
async def get_memory_summary(
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """获取记忆摘要"""
    try:
        if not memory_service.is_enabled():
            return MemorySummaryResponse(
                success=True,
                message="记忆服务未启用",
                enabled=False
            )
        
        summary_data = {
            "enabled": True,
            "success": True,
            "message": "记忆摘要获取成功"
        }
        
        recent_memories = []
        
        # 获取用户记忆摘要
        if user_id:
            user_memories = await memory_service.get_all_user_memories(user_id, limit=3)
            summary_data["user_memories"] = len(user_memories)
            recent_memories.extend(user_memories[:2])
        
        # 获取Agent记忆摘要
        if agent_id:
            agent_memories = await memory_service.get_all_agent_memories(agent_id, limit=3)
            summary_data["agent_memories"] = len(agent_memories)
            recent_memories.extend(agent_memories[:2])
        
        summary_data["recent_memories"] = recent_memories[:5]
        
        return MemorySummaryResponse(**summary_data)
        
    except Exception as e:
        app_logger.error(f"获取记忆摘要失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/status")
async def get_memory_status():
    """获取记忆服务状态"""
    try:
        return {
            "success": True,
            "enabled": memory_service.is_enabled(),
            "service": "mem0",
            "message": "记忆服务状态正常" if memory_service.is_enabled() else "记忆服务未启用"
        }
        
    except Exception as e:
        app_logger.error(f"获取记忆服务状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 