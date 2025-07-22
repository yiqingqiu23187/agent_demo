from typing import List, Dict, Any, Optional
import uuid
from mem0 import Memory
from app.core.config import settings
from app.utils.logger import app_logger


class MemoryService:
    """Mem0记忆管理服务"""
    
    def __init__(self):
        self.memory = None
        self._initialize_memory()
    
    def _initialize_memory(self):
        """初始化Mem0实例"""
        try:
            if not settings.openai_api_key:
                app_logger.warning("OpenAI API密钥未配置，记忆功能将被禁用")
                return
                
            # 创建数据目录
            import os
            os.makedirs("./data", exist_ok=True)
            
            # 设置OpenAI API密钥环境变量
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
            
            from mem0 import Memory
            
            # 使用简化配置，依赖默认行为
            config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "path": "./data/mem0_db"
                    }
                }
            }
            
            # 尝试用配置初始化，如果失败则使用默认配置
            try:
                self.memory = Memory.from_config(config)
            except Exception as config_error:
                app_logger.warning(f"使用配置初始化失败: {config_error}，尝试默认配置")
                # 使用默认配置
                self.memory = Memory()
                
            app_logger.info("Mem0记忆管理初始化成功")
                
        except Exception as e:
            app_logger.error(f"Mem0初始化失败: {e}")
            app_logger.info("记忆功能将被禁用，但不影响其他功能")
            self.memory = None
    
    def is_enabled(self) -> bool:
        """检查记忆服务是否启用"""
        return self.memory is not None
    
    async def add_user_memory(
        self,
        messages: List[Dict[str, str]], 
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """添加用户记忆"""
        if not self.is_enabled():
            return {"success": False, "message": "记忆服务未启用"}
        
        try:
            result = self.memory.add(
                messages=messages,
                user_id=user_id,
                metadata=metadata or {}
            )
            
            app_logger.info(f"成功添加用户 {user_id} 的记忆")
            return {
                "success": True,
                "message": "用户记忆添加成功",
                "memories": result
            }
            
        except Exception as e:
            app_logger.error(f"添加用户记忆失败: {e}")
            return {"success": False, "message": str(e)}
    
    async def add_agent_memory(
        self,
        messages: List[Dict[str, str]], 
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """添加Agent记忆"""
        if not self.is_enabled():
            return {"success": False, "message": "记忆服务未启用"}
        
        try:
            result = self.memory.add(
                messages=messages,
                agent_id=agent_id,
                metadata=metadata or {}
            )
            
            app_logger.info(f"成功添加Agent {agent_id} 的记忆")
            return {
                "success": True,
                "message": "Agent记忆添加成功", 
                "memories": result
            }
            
        except Exception as e:
            app_logger.error(f"添加Agent记忆失败: {e}")
            return {"success": False, "message": str(e)}
    
    async def add_session_memory(
        self,
        messages: List[Dict[str, str]], 
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """添加会话记忆"""
        if not self.is_enabled():
            return {"success": False, "message": "记忆服务未启用"}
        
        try:
            result = self.memory.add(
                messages=messages,
                run_id=session_id,
                metadata=metadata or {}
            )
            
            app_logger.info(f"成功添加会话 {session_id} 的记忆")
            return {
                "success": True,
                "message": "会话记忆添加成功",
                "memories": result
            }
            
        except Exception as e:
            app_logger.error(f"添加会话记忆失败: {e}")
            return {"success": False, "message": str(e)}
    
    async def search_user_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索用户记忆"""
        if not self.is_enabled():
            return []
        
        try:
            result = self.memory.search(
                query=query,
                user_id=user_id,
                limit=limit
            )
            
            return result.get("results", [])
            
        except Exception as e:
            app_logger.error(f"搜索用户记忆失败: {e}")
            return []
    
    async def search_agent_memories(
        self,
        query: str,
        agent_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索Agent记忆"""
        if not self.is_enabled():
            return []
        
        try:
            result = self.memory.search(
                query=query,
                agent_id=agent_id,
                limit=limit
            )
            
            return result.get("results", [])
            
        except Exception as e:
            app_logger.error(f"搜索Agent记忆失败: {e}")
            return []
    
    async def search_session_memories(
        self,
        query: str,
        session_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索会话记忆"""
        if not self.is_enabled():
            return []
        
        try:
            result = self.memory.search(
                query=query,
                run_id=session_id,
                limit=limit
            )
            
            return result.get("results", [])
            
        except Exception as e:
            app_logger.error(f"搜索会话记忆失败: {e}")
            return []
    
    async def get_all_user_memories(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取用户所有记忆"""
        if not self.is_enabled():
            return []
        
        try:
            result = self.memory.get_all(
                user_id=user_id,
                limit=limit
            )
            
            return result
            
        except Exception as e:
            app_logger.error(f"获取用户记忆失败: {e}")
            return []
    
    async def get_all_agent_memories(
        self,
        agent_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取Agent所有记忆"""
        if not self.is_enabled():
            return []
        
        try:
            result = self.memory.get_all(
                agent_id=agent_id,
                limit=limit
            )
            
            return result
            
        except Exception as e:
            app_logger.error(f"获取Agent记忆失败: {e}")
            return []
    
    async def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """删除指定记忆"""
        if not self.is_enabled():
            return {"success": False, "message": "记忆服务未启用"}
        
        try:
            self.memory.delete(memory_id=memory_id)
            
            return {
                "success": True,
                "message": f"记忆 {memory_id} 删除成功"
            }
            
        except Exception as e:
            app_logger.error(f"删除记忆失败: {e}")
            return {"success": False, "message": str(e)}
    
    async def delete_all_user_memories(self, user_id: str) -> Dict[str, Any]:
        """删除用户所有记忆"""
        if not self.is_enabled():
            return {"success": False, "message": "记忆服务未启用"}
        
        try:
            self.memory.delete_all(user_id=user_id)
            
            return {
                "success": True,
                "message": f"用户 {user_id} 的所有记忆删除成功"
            }
            
        except Exception as e:
            app_logger.error(f"删除用户记忆失败: {e}")
            return {"success": False, "message": str(e)}
    
    async def delete_all_session_memories(self, session_id: str) -> Dict[str, Any]:
        """删除会话所有记忆"""
        if not self.is_enabled():
            return {"success": False, "message": "记忆服务未启用"}
        
        try:
            self.memory.delete_all(run_id=session_id)
            
            return {
                "success": True,
                "message": f"会话 {session_id} 的所有记忆删除成功"
            }
            
        except Exception as e:
            app_logger.error(f"删除会话记忆失败: {e}")
            return {"success": False, "message": str(e)}
    
    def reset_all_memories(self) -> Dict[str, Any]:
        """重置所有记忆"""
        if not self.is_enabled():
            return {"success": False, "message": "记忆服务未启用"}
        
        try:
            self.memory.reset()
            
            return {
                "success": True,
                "message": "所有记忆重置成功"
            }
            
        except Exception as e:
            app_logger.error(f"重置记忆失败: {e}")
            return {"success": False, "message": str(e)}


# 创建全局记忆服务实例
memory_service = MemoryService() 