from typing import List, Dict, Any, Optional
import time
import asyncio
import httpx
from app.core.memory_service import memory_service  
from app.core.monitoring_service import monitoring_service
from app.utils.logger import app_logger
from app.core.config import settings


class DifyAgent:
    """Dify平台Agent - 连接Dify工作流的智能体"""
    
    def __init__(self):
        self.memory_service = memory_service
        self.monitoring_service = monitoring_service
        self.agent_type = "dify"
        # 这里可以配置Dify API的相关信息
        self.dify_api_url = getattr(settings, 'dify_api_url', None)
        self.dify_api_key = getattr(settings, 'dify_api_key', None)
    
    async def execute(
        self,
        task: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        use_memory: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """执行Dify Agent任务"""
        start_time = time.time()
        
        try:
            app_logger.info(f"开始执行Dify Agent任务: {task}")
            
            # 检索相关记忆
            relevant_memories = await self._retrieve_memories(
                task, user_id, agent_id, session_id, use_memory
            )
            
            # 构建输入参数
            dify_input = {
                "task": task,
                "user_id": user_id,
                "session_id": session_id,
                "memory_context": self._build_memory_context(relevant_memories)
            }
            dify_input.update(kwargs)  # 添加额外参数
            
            # 调用Dify API
            if self.dify_api_url and self.dify_api_key:
                result = await self._call_dify_api(dify_input, workflow_id)
            else:
                # 模拟Dify调用（用于演示）
                result = await self._simulate_dify_call(dify_input)
            
            execution_time = time.time() - start_time
            
            # 保存新的记忆
            await self._save_memories(
                task, result, user_id, agent_id, session_id, 
                use_memory
            )
            
            app_logger.info(f"Dify Agent任务执行完成，耗时: {execution_time:.2f}秒")
            
            return {
                "success": True,
                "result": result,
                "agent_type": self.agent_type,
                "tools_used": ["dify_workflow"],
                "execution_time": execution_time,
                "memory_used": len(relevant_memories),
                "workflow_id": workflow_id
            }
            
        except Exception as e:
            app_logger.error(f"Dify Agent执行失败: {e}")
            return {
                "success": False,
                "result": f"Dify Agent执行失败: {str(e)}",
                "agent_type": self.agent_type,
                "tools_used": [],
                "execution_time": time.time() - start_time,
                "memory_used": 0,
                "error": str(e)
            }
        finally:
            # 刷新监控数据
            self.monitoring_service.flush()
    
    async def _call_dify_api(
        self, 
        input_data: Dict[str, Any], 
        workflow_id: Optional[str] = None
    ) -> str:
        """调用真实的Dify API"""
        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.dify_api_key}",
                    "Content-Type": "application/json"
                }
                
                # 构建API请求URL
                if workflow_id:
                    url = f"{self.dify_api_url}/workflows/{workflow_id}/run"
                else:
                    url = f"{self.dify_api_url}/workflows/run"
                
                # 构建请求体
                request_body = {
                    "inputs": input_data,
                    "response_mode": "blocking",  # 或 "streaming"
                    "user": input_data.get("user_id", "anonymous")
                }
                
                # 发送请求
                response = await client.post(
                    url,
                    headers=headers,
                    json=request_body,
                    timeout=30.0
                )
                
                response.raise_for_status()
                result_data = response.json()
                
                # 提取结果
                if "data" in result_data and "outputs" in result_data["data"]:
                    return result_data["data"]["outputs"].get("result", "Dify处理完成")
                else:
                    return f"Dify工作流执行完成: {result_data}"
                
        except httpx.RequestError as e:
            app_logger.error(f"Dify API请求失败: {e}")
            raise Exception(f"Dify API连接失败: {str(e)}")
        except httpx.HTTPStatusError as e:
            app_logger.error(f"Dify API返回错误: {e.response.status_code}")
            raise Exception(f"Dify API错误: {e.response.status_code}")
        except Exception as e:
            app_logger.error(f"Dify API调用异常: {e}")
            raise
    
    async def _simulate_dify_call(self, input_data: Dict[str, Any]) -> str:
        """模拟Dify API调用（用于演示）"""
        # 模拟API调用延迟
        await asyncio.sleep(0.5)
        
        task = input_data.get("task", "")
        memory_context = input_data.get("memory_context", "")
        
        # 根据任务类型生成不同的模拟响应
        if "天气" in task:
            result = f"基于Dify平台的天气查询结果：{task} - 今天天气晴朗，温度适宜。"
        elif "计算" in task:
            result = f"基于Dify平台的计算结果：{task} - 计算已完成。"
        elif "搜索" in task:
            result = f"基于Dify平台的搜索结果：{task} - 已找到相关信息。"
        else:
            result = f"基于Dify平台的执行结果：已完成任务 '{task}'"
        
        # 如果有记忆上下文，在结果中体现
        if memory_context:
            result += f"\n(结合了历史记忆信息)"
        
        return result
    
    async def _retrieve_memories(
        self, 
        task: str, 
        user_id: Optional[str], 
        agent_id: Optional[str], 
        session_id: Optional[str], 
        use_memory: bool
    ) -> List[Dict[str, Any]]:
        """检索相关记忆"""
        relevant_memories = []
        
        if not use_memory or not self.memory_service.is_enabled():
            return relevant_memories
        
        try:
            if user_id:
                user_memories = await self.memory_service.search_user_memories(
                    query=task, user_id=user_id, limit=3
                )
                relevant_memories.extend(user_memories)
            
            if agent_id:
                agent_memories = await self.memory_service.search_agent_memories(
                    query=task, agent_id=agent_id, limit=3
                )
                relevant_memories.extend(agent_memories)
            
            if session_id:
                session_memories = await self.memory_service.search_session_memories(
                    query=task, session_id=session_id, limit=3
                )
                relevant_memories.extend(session_memories)
                
        except Exception as e:
            app_logger.warning(f"记忆检索失败: {e}")
        
        return relevant_memories
    
    def _build_memory_context(self, relevant_memories: List[Dict[str, Any]]) -> str:
        """构建记忆上下文"""
        if not relevant_memories:
            return ""
        
        memory_items = [f"- {mem.get('memory', '')}" for mem in relevant_memories[:5]]
        return f"相关历史记忆:\n" + "\n".join(memory_items)
    
    async def _save_memories(
        self, 
        task: str, 
        result: str, 
        user_id: Optional[str], 
        agent_id: Optional[str], 
        session_id: Optional[str],
        use_memory: bool
    ):
        """保存新的记忆"""
        if not use_memory or not self.memory_service.is_enabled():
            return
        
        try:
            messages = [
                {"role": "user", "content": task},
                {"role": "assistant", "content": result}
            ]
            
            metadata = {"task_type": "dify_agent", "platform": "dify"}
            
            if user_id:
                await self.memory_service.add_user_memory(
                    messages=messages,
                    user_id=user_id,
                    metadata=metadata
                )
            
            if agent_id:
                await self.memory_service.add_agent_memory(
                    messages=messages,
                    agent_id=agent_id,
                    metadata=metadata
                )
            
            if session_id:
                await self.memory_service.add_session_memory(
                    messages=messages,
                    session_id=session_id,
                    metadata=metadata
                )
        except Exception as e:
            app_logger.warning(f"保存记忆失败: {e}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """获取Agent信息"""
        return {
            "agent_type": self.agent_type,
            "name": "Dify Agent",
            "description": "连接Dify平台工作流的智能体，支持复杂的业务流程",
            "supports_tools": True,  # 通过Dify工作流支持工具
            "supports_memory": True,
            "requires_dify_config": True,
            "dify_configured": bool(self.dify_api_url and self.dify_api_key),
            "api_url": self.dify_api_url if self.dify_api_url else "未配置"
        }
    
    async def list_workflows(self) -> List[Dict[str, Any]]:
        """列出可用的Dify工作流（如果配置了API）"""
        if not (self.dify_api_url and self.dify_api_key):
            return [{"id": "demo", "name": "演示工作流", "description": "用于演示的模拟工作流"}]
        
        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.dify_api_key}",
                    "Content-Type": "application/json"
                }
                
                response = await client.get(
                    f"{self.dify_api_url}/workflows",
                    headers=headers,
                    timeout=10.0
                )
                
                response.raise_for_status()
                data = response.json()
                
                # 返回工作流列表
                return data.get("workflows", [])
                
        except Exception as e:
            app_logger.error(f"获取Dify工作流列表失败: {e}")
            return []


# 创建全局Dify Agent实例
dify_agent = DifyAgent() 