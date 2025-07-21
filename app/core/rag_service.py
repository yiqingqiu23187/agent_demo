from typing import List, Dict, Any, Optional
import asyncio
from app.core.vector_service import vector_service
from app.utils.logger import app_logger


class RAGService:
    """检索增强生成服务 - 只负责知识检索和知识库管理"""
    
    def __init__(self):
        self.vector_service = vector_service
    
    async def retrieve_documents(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """检索相关文档"""
        try:
            app_logger.info(f"开始检索相关文档: {query}")
            retrieved_docs = await self.vector_service.search(
                query=query,
                collection_name=collection_name,
                top_k=top_k
            )
            
            if not retrieved_docs:
                app_logger.warning("未找到相关文档")
                return {
                    "documents": [],
                    "context": "",
                    "query": query,
                    "collection_name": collection_name,
                    "retrieved_count": 0
                }
            
            # 过滤低分文档（如果设置了阈值）
            if score_threshold > 0:
                filtered_docs = [doc for doc in retrieved_docs if doc.get("score", 0) >= score_threshold]
                if filtered_docs:
                    retrieved_docs = filtered_docs
            
            # 构建上下文
            context = self._build_context(retrieved_docs)
            
            result = {
                "documents": retrieved_docs,
                "context": context,
                "sources": self._format_sources(retrieved_docs),
                "query": query,
                "collection_name": collection_name,
                "retrieved_count": len(retrieved_docs)
            }
            
            app_logger.info(f"成功检索到 {len(retrieved_docs)} 个相关文档")
            return result
            
        except Exception as e:
            app_logger.error(f"文档检索失败: {e}")
            raise
    
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
                "collection_name": collection_name,
                "added_count": len(texts)
            }
            
        except Exception as e:
            app_logger.error(f"添加文档到知识库失败: {e}")
            raise
    
    async def batch_add_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: str
    ) -> Dict[str, Any]:
        """批量添加文档"""
        try:
            texts = []
            metadatas = []
            
            for doc in documents:
                texts.append(doc.get("content", ""))
                metadatas.append(doc.get("metadata", {}))
            
            return await self.add_documents_to_knowledge_base(
                texts=texts,
                collection_name=collection_name,
                metadatas=metadatas
            )
            
        except Exception as e:
            app_logger.error(f"批量添加文档失败: {e}")
            raise
    
    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """构建上下文文本"""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            
            # 如果有元数据中的标题信息，添加到上下文中
            title = metadata.get('title', '')
            if title:
                context_parts.append(f"文档{i} [{title}]: {content}")
            else:
                context_parts.append(f"文档{i}: {content}")
        
        return "\n\n".join(context_parts)
    
    def _format_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化来源信息"""
        sources = []
        for doc in retrieved_docs:
            content = doc.get("content", "")
            source = {
                "content": content[:200] + "..." if len(content) > 200 else content,
                "metadata": doc.get("metadata", {}),
                "relevance_score": doc.get("score", 0)
            }
            sources.append(source)
        
        return sources
    
    async def search_documents_by_metadata(
        self,
        collection_name: str,
        metadata_filter: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """根据元数据搜索文档"""
        try:
            # 这里需要vector_service支持元数据过滤功能
            # 暂时返回空列表，实际实现需要根据具体的向量数据库实现
            app_logger.info(f"根据元数据搜索文档: {metadata_filter}")
            
            # TODO: 实现基于元数据的文档搜索
            return []
            
        except Exception as e:
            app_logger.error(f"元数据搜索失败: {e}")
            raise
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """获取知识库信息"""
        try:
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
    
    async def get_document_by_id(
        self,
        collection_name: str,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        try:
            # TODO: 实现根据ID获取文档的功能
            app_logger.info(f"获取文档: {document_id}")
            return None
            
        except Exception as e:
            app_logger.error(f"获取文档失败: {e}")
            raise
    
    async def update_document(
        self,
        collection_name: str,
        document_id: str,
        new_content: str,
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """更新文档"""
        try:
            # TODO: 实现文档更新功能
            app_logger.info(f"更新文档: {document_id}")
            
            return {
                "success": False,
                "message": "文档更新功能暂未实现"
            }
            
        except Exception as e:
            app_logger.error(f"更新文档失败: {e}")
            raise
    
    async def delete_document(
        self,
        collection_name: str,
        document_id: str
    ) -> Dict[str, Any]:
        """删除文档"""
        try:
            # TODO: 实现文档删除功能
            app_logger.info(f"删除文档: {document_id}")
            
            return {
                "success": False,
                "message": "文档删除功能暂未实现"
            }
            
        except Exception as e:
            app_logger.error(f"删除文档失败: {e}")
            raise


# 创建全局RAG服务实例
rag_service = RAGService() 