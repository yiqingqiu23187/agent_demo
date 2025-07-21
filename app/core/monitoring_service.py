from typing import Optional, Dict, Any, List
import os
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from app.core.config import settings
from app.utils.logger import app_logger


class MonitoringService:
    """LangFuse监控服务 - 使用官方集成方式"""
    
    def __init__(self):
        self.enabled = False
        self._initialize_langfuse()
    
    def _initialize_langfuse(self):
        """初始化LangFuse客户端"""
        try:
            if settings.langfuse_secret_key and settings.langfuse_public_key:
                # 设置环境变量，供LangFuse SDK使用
                os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
                os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
                os.environ["LANGFUSE_HOST"] = settings.langfuse_host
                
                # 初始化LangFuse客户端（使用单例模式）
                Langfuse(
                    secret_key=settings.langfuse_secret_key,
                    public_key=settings.langfuse_public_key,
                    host=settings.langfuse_host
                )
                
                self.enabled = True
                app_logger.info("LangFuse监控初始化成功")
            else:
                app_logger.warning("LangFuse配置未完整，监控功能将被禁用")
        except Exception as e:
            app_logger.error(f"LangFuse初始化失败: {e}")
            self.enabled = False
    
    def is_enabled(self) -> bool:
        """检查监控是否启用"""
        return self.enabled
    
    def get_langchain_callback_handler(self) -> Optional[CallbackHandler]:
        """获取LangChain回调处理器"""
        if not self.is_enabled():
            return None
        
        try:
            return CallbackHandler()
        except Exception as e:
            app_logger.error(f"创建LangChain回调处理器失败: {e}")
            return None
    
    def get_client(self):
        """获取LangFuse客户端"""
        if not self.is_enabled():
            return None
        
        try:
            return get_client()
        except Exception as e:
            app_logger.error(f"获取LangFuse客户端失败: {e}")
            return None
    
    def flush(self):
        """刷新缓存的数据到LangFuse"""
        if not self.is_enabled():
            return
        
        try:
            client = self.get_client()
            if client:
                client.flush()
        except Exception as e:
            app_logger.error(f"刷新LangFuse数据失败: {e}")
    
    def shutdown(self):
        """关闭LangFuse连接"""
        if not self.is_enabled():
            return
        
        try:
            client = self.get_client()
            if client:
                client.shutdown()
        except Exception as e:
            app_logger.error(f"关闭LangFuse连接失败: {e}")
    
    # 保留一些便捷方法用于直接操作
    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """创建调用链追踪"""
        if not self.is_enabled():
            return None
        
        try:
            client = self.get_client()
            if not client:
                return None
            
            trace = client.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {}
            )
            return trace.id
        except Exception as e:
            app_logger.error(f"创建trace失败: {e}")
            return None
    
    def score_trace(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None
    ):
        """为trace评分"""
        if not self.is_enabled():
            return
        
        try:
            client = self.get_client()
            if not client:
                return
            
            client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment
            )
        except Exception as e:
            app_logger.error(f"为trace评分失败: {e}")


# 创建全局监控服务实例
monitoring_service = MonitoringService() 