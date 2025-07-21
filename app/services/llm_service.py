from typing import Optional, AsyncGenerator, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from app.core.config import settings
from app.utils.logger import app_logger


class LLMService:
    """大语言模型服务"""
    
    def __init__(self):
        self._models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化模型实例"""
        try:
            # OpenAI模型
            if settings.openai_api_key:
                self._models["gpt-4o"] = ChatOpenAI(
                    model="gpt-4o",
                    api_key=settings.openai_api_key,
                    temperature=0.7
                )
                self._models["gpt-4o-mini"] = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=settings.openai_api_key,
                    temperature=0.7
                )
                app_logger.info("OpenAI模型初始化成功")
            
            # Anthropic模型
            if settings.anthropic_api_key:
                self._models["claude-3-5-sonnet-20241022"] = ChatAnthropic(
                    model="claude-3-5-sonnet-20241022",
                    api_key=settings.anthropic_api_key,
                    temperature=0.7
                )
                self._models["claude-3-haiku-20240307"] = ChatAnthropic(
                    model="claude-3-haiku-20240307",
                    api_key=settings.anthropic_api_key,
                    temperature=0.7
                )
                app_logger.info("Anthropic模型初始化成功")
                
        except Exception as e:
            app_logger.error(f"模型初始化失败: {e}")
    
    def get_available_models(self) -> list:
        """获取可用模型列表"""
        return list(self._models.keys())
    
    async def chat_completion(
        self,
        message: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """聊天补全"""
        try:
            if model not in self._models:
                raise ValueError(f"模型 {model} 不可用")
            
            llm = self._models[model]
            if temperature != 0.7:
                llm.temperature = temperature
            if max_tokens:
                llm.max_tokens = max_tokens
            
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=message))
            
            response = await llm.ainvoke(messages)
            
            return {
                "content": response.content,
                "model": model,
                "usage": getattr(response, 'usage_metadata', {}),
                "finish_reason": getattr(response, 'response_metadata', {}).get('finish_reason')
            }
            
        except Exception as e:
            app_logger.error(f"聊天补全失败: {e}")
            raise
    
    async def stream_chat_completion(
        self,
        message: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """流式聊天补全"""
        try:
            if model not in self._models:
                raise ValueError(f"模型 {model} 不可用")
            
            llm = self._models[model]
            if temperature != 0.7:
                llm.temperature = temperature
            
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=message))
            
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            app_logger.error(f"流式聊天补全失败: {e}")
            raise
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """获取模型信息"""
        if model not in self._models:
            return {"error": f"模型 {model} 不可用"}
        
        model_info = {
            "name": model,
            "provider": "openai" if "gpt" in model else "anthropic",
            "available": True
        }
        
        return model_info


# 创建全局LLM服务实例
llm_service = LLMService() 