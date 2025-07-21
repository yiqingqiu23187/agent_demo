from typing import List, Dict, Any, Optional
import asyncio
from app.services.llm_service import llm_service
from app.services.vector_service import vector_service
from app.services.monitoring_service import monitoring_service
from app.utils.logger import app_logger
from langchain_core.messages import HumanMessage, SystemMessage


class RAGService:
    """检索增强生成服务"""
    
    def __init__(self):
        self.llm_service = llm_service
        self.vector_service = vector_service
        self.monitoring_service = monitoring_service
    
    async def query(
        self,
        query: str,
        collection_name: str,
        model: str = "gpt-4o-mini",
        top_k: int = 5,
        score_threshold: float = 0.7,
        temperature: float = 0.7,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """RAG查询"""
        try:
            # 获取LangFuse回调处理器
            langfuse_handler = self.monitoring_service.get_langchain_callback_handler()
            callbacks = [langfuse_handler] if langfuse_handler else []
            
            # 1. 检索相关文档
            app_logger.info(f"开始检索相关文档: {query}")
            retrieved_docs = await self.vector_service.search(
                query=query,
                collection_name=collection_name,
                top_k=top_k
            )
            
            if not retrieved_docs:
                app_logger.warning("未找到相关文档")
                return {
                    "answer": "抱歉，我在知识库中没有找到相关信息来回答您的问题。",
                    "sources": [],
                    "query": query,
                    "collection_name": collection_name
                }
            
            # 2. 构建上下文
            context = self._build_context(retrieved_docs)
            
            # 3. 生成提示词
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(query, context)
            
            # 4. 调用LLM生成答案（通过LangChain，自动追踪到LangFuse）
            app_logger.info(f"调用LLM生成答案: {model}")
            
            # 检查模型可用性
            if model not in self.llm_service._models:
                raise ValueError(f"模型 {model} 不可用")
            
            llm = self.llm_service._models[model]
            
            # 构建消息
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await llm.ainvoke(
                messages,
                config={
                    "callbacks": callbacks,
                    "metadata": {
                        "langfuse_user_id": user_id,
                        "langfuse_session_id": session_id,
                        "langfuse_tags": ["rag-query"],
                        "query": query,
                        "collection_name": collection_name,
                        "model": model,
                        "top_k": top_k,
                        "retrieved_docs_count": len(retrieved_docs)
                    }
                }
            )
            
            # 5. 格式化响应
            result = {
                "answer": response.content,
                "sources": self._format_sources(retrieved_docs),
                "query": query,
                "collection_name": collection_name,
                "model_used": model,
                "retrieved_docs_count": len(retrieved_docs),
                "trace_id": langfuse_handler.last_trace_id if langfuse_handler else None
            }
            
            app_logger.info("RAG查询完成")
            return result
            
        except Exception as e:
            app_logger.error(f"RAG查询失败: {e}")
            raise
        finally:
            # 刷新监控数据
            self.monitoring_service.flush()
    
    async def add_documents_to_knowledge_base(
        self,
        texts: List[str],
        collection_name: str,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """添加文档到知识库"""
        try:
            app_logger.info(f"添加文档到知识库: {collection_name}")
            
            document_ids = await self.vector_service.add_texts(
                texts=texts,
                collection_name=collection_name,
                metadatas=metadatas
            )
            
            return {
                "success": True,
                "message": f"成功添加 {len(texts)} 个文档到知识库 {collection_name}",
                "document_ids": document_ids,
                "collection_name": collection_name
            }
            
        except Exception as e:
            app_logger.error(f"添加文档到知识库失败: {e}")
            raise
    
    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """构建上下文文本"""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"文档{i}：{doc['content']}")
        
        return "\n\n".join(context_parts)
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个专业的AI助手。请基于提供的上下文信息来回答用户的问题。

回答要求：
1. 答案必须基于提供的上下文信息
2. 如果上下文中没有足够信息回答问题，请明确说明
3. 保持回答的准确性和相关性
4. 回答要简洁明了，避免冗余
5. 可以引用具体的文档内容来支持你的回答"""
    
    def _build_user_prompt(self, query: str, context: str) -> str:
        """构建用户提示词"""
        return f"""基于以下上下文信息，请回答用户的问题：

上下文信息：
{context}

用户问题：{query}

请提供准确、有用的回答："""
    
    def _format_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化来源信息"""
        sources = []
        for doc in retrieved_docs:
            source = {
                "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                "metadata": doc["metadata"],
                "relevance_score": doc["score"]
            }
            sources.append(source)
        
        return sources
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """获取知识库信息"""
        try:
            # 这里可以添加获取知识库统计信息的逻辑
            # 目前返回基本信息
            return {
                "collection_name": collection_name,
                "vector_store_type": self.vector_service.vector_store_type,
                "status": "active"
            }
        except Exception as e:
            app_logger.error(f"获取知识库信息失败: {e}")
            raise
    
    async def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """删除知识库"""
        try:
            success = await self.vector_service.delete_collection(collection_name)
            
            if success:
                return {
                    "success": True,
                    "message": f"成功删除知识库: {collection_name}"
                }
            else:
                return {
                    "success": False,
                    "message": f"删除知识库失败: {collection_name}"
                }
                
        except Exception as e:
            app_logger.error(f"删除知识库失败: {e}")
            raise


# 创建全局RAG服务实例
rag_service = RAGService() 