from typing import Dict, Any, Optional, List
import time
import httpx
from app.utils.logger import app_logger
from app.core.config import settings


class DifyAgent:
    """Dify平台Agent - 简单的API调用封装"""
    
    def __init__(self):
        self.agent_type = "dify"
        # 使用配置中的Dify API信息
        self.dify_api_url = getattr(settings, 'dify_api_base_url', None)
        self.dify_api_key = getattr(settings, 'dify_api_key', None)
    
    async def execute(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """执行Dify Agent任务"""
        start_time = time.time()
        
        try:
            app_logger.info(f"开始执行Dify Agent任务: {query}")
            
            # 检查配置
            if not (self.dify_api_url and self.dify_api_key):
                raise Exception("Dify API配置不完整，请检查DIFY_API_BASE_URL和DIFY_API_KEY环境变量")
            
            # 调用Dify chatflow API
            result = await self._call_dify_chatflow_api(
                query=query,
                conversation_id=conversation_id,
                user_id=user_id,
                inputs=inputs or {}
            )
            
            execution_time = time.time() - start_time
            
            app_logger.info(f"Dify Agent任务执行完成，耗时: {execution_time:.2f}秒")
            
            return {
                "success": True,
                "result": result,
                "agent_type": self.agent_type,
                "execution_time": execution_time,
                "conversation_id": result.get("conversation_id") if isinstance(result, dict) else None
            }
            
        except Exception as e:
            app_logger.error(f"Dify Agent执行失败: {e}")
            return {
                "success": False,
                "result": f"Dify Agent执行失败: {str(e)}",
                "agent_type": self.agent_type,
                "execution_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def _call_dify_chatflow_api(
        self, 
        query: str, 
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        inputs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """调用Dify chatflow API"""
        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.dify_api_key}",
                    "Content-Type": "application/json"
                }
                
                # 构建API请求URL - 使用chatflow的chat-messages端点
                url = f"{self.dify_api_url}/chat-messages"
                
                # 为蓝领招聘chatflow准备固定的inputs参数
                default_inputs = {
                    "gender": "男",
                    "resume_id": "00032da5-f475-40cb-8de5-556827b70f20",
                    "job_id": "00393b28-431f-4f16-b9e7-451db6645e8c"
                }
                
                # 合并用户传入的inputs和默认inputs
                final_inputs = {**default_inputs, **(inputs or {})}
                
                # 构建请求体
                request_body = {
                    "query": query,
                    "inputs": final_inputs,
                    "response_mode": "blocking",  # 使用阻塞模式
                    "user": user_id or "anonymous"
                }
                
                # 如果有conversation_id，添加到请求中以继续对话
                if conversation_id:
                    request_body["conversation_id"] = conversation_id
                
                app_logger.info(f"调用Dify API: {url}")
                app_logger.debug(f"请求体: {request_body}")
                
                # 发送请求
                response = await client.post(
                    url,
                    headers=headers,
                    json=request_body,
                    timeout=30.0
                )
                
                response.raise_for_status()
                result_data = response.json()
                
                app_logger.info(f"Dify API响应状态码: {response.status_code}")
                app_logger.debug(f"Dify API响应: {result_data}")
                
                # 返回完整的响应数据
                return result_data
                
        except httpx.RequestError as e:
            app_logger.error(f"Dify API请求失败: {e}")
            raise Exception(f"Dify API连接失败: {str(e)}")
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_body = e.response.json()
                error_detail = f" - {error_body}"
            except:
                pass
            app_logger.error(f"Dify API返回错误: {e.response.status_code}{error_detail}")
            raise Exception(f"Dify API错误: {e.response.status_code}{error_detail}")
        except Exception as e:
            app_logger.error(f"Dify API调用异常: {e}")
            raise
    
    async def list_workflows(self) -> List[Dict[str, Any]]:
        """获取Dify工作流列表"""
        try:
            if not (self.dify_api_url and self.dify_api_key):
                raise Exception("Dify API配置不完整")
            
            # 这是一个示例实现，实际的Dify API可能不同
            # 由于我们使用的是chatflow，返回一个示例工作流列表
            return [
                {
                    "id": "blue_collar_recruitment",
                    "name": "蓝领招聘助手",
                    "description": "专门用于蓝领招聘场景的对话流程",
                    "status": "published"
                }
            ]
        except Exception as e:
            app_logger.error(f"获取Dify工作流列表失败: {e}")
            return []
    
    def get_agent_info(self) -> Dict[str, Any]:
        """获取Agent信息"""
        return {
            "agent_type": self.agent_type,
            "name": "Dify Chatflow Agent",
            "description": "连接Dify平台的蓝领招聘Chatflow智能体，专门用于蓝领求职场景的对话式交互。",
            "supports_tools": True,  # 由Dify平台提供
            "supports_memory": True,  # 由Dify平台提供
            "supports_rag": True,  # 由Dify平台提供
            "requires_dify_config": True,
            "dify_configured": bool(self.dify_api_url and self.dify_api_key),
            "api_url": self.dify_api_url if self.dify_api_url else "未配置",
            "chatflow_type": "chat-messages",
            "scenario": "蓝领招聘",
            "fixed_inputs": {
                "gender": "男",
                "resume_id": "00032da5-f475-40cb-8de5-556827b70f20",
                "job_id": "00393b28-431f-4f16-b9e7-451db6645e8c"
            }
        }
    
    async def create_conversation(
        self, 
        initial_message: str,
        user_id: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建新对话（不指定conversation_id）"""
        return await self.execute(
            query=initial_message,
            user_id=user_id,
            conversation_id=None,  # 不指定conversation_id以创建新对话
            inputs=inputs
        )
    
    async def continue_conversation(
        self,
        message: str,
        conversation_id: str,
        user_id: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """继续现有对话"""
        return await self.execute(
            query=message,
            user_id=user_id,
            conversation_id=conversation_id,
            inputs=inputs
        )


# 创建全局Dify Agent实例
dify_agent = DifyAgent() 