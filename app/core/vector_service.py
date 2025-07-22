from typing import List, Dict, Any, Optional
import os
from abc import ABC, abstractmethod
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
import chromadb
from app.core.config import settings
from app.utils.logger import app_logger


class VectorStoreBase(ABC):
    """向量存储基类"""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document], collection_name: str) -> List[str]:
        pass
    
    @abstractmethod
    async def similarity_search(self, query: str, collection_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        pass


class ChromaVectorStore(VectorStoreBase):
    """Chroma向量存储实现"""
    
    def __init__(self):
        try:
            # 尝试使用OpenAI embeddings（如果API密钥可用且有配额）
            if settings.openai_api_key:
                try:
                    # 测试OpenAI API是否可用
                    test_embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
                    # 进行一个小测试
                    test_embeddings.embed_query("test")
                    self.embeddings = test_embeddings
                    app_logger.info("使用OpenAI Embeddings")
                except Exception as e:
                    app_logger.warning(f"OpenAI Embeddings不可用，切换到本地模型: {e}")
                    # 使用本地HuggingFace模型
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'}
                    )
                    app_logger.info("使用本地HuggingFace Embeddings")
            else:
                # 直接使用本地模型
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2", 
                    model_kwargs={'device': 'cpu'}
                )
                app_logger.info("使用本地HuggingFace Embeddings")
        except Exception as e:
            app_logger.error(f"Embedding模型初始化失败: {e}")
            # 如果都失败了，使用一个简单的fallback
            raise RuntimeError("无法初始化任何embedding模型")
            
        self.persist_directory = settings.chroma_persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
    
    async def add_documents(self, documents: List[Document], collection_name: str) -> List[str]:
        """添加文档到向量存储"""
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            ids = vectorstore.add_documents(documents)
            vectorstore.persist()
            
            app_logger.info(f"成功添加 {len(documents)} 个文档到 {collection_name}")
            return ids
            
        except Exception as e:
            app_logger.error(f"添加文档失败: {e}")
            raise
    
    async def similarity_search(self, query: str, collection_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """相似度搜索"""
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            results = vectorstore.similarity_search_with_score(query, k=top_k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            return formatted_results
            
        except Exception as e:
            app_logger.error(f"相似度搜索失败: {e}")
            raise
    
    async def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        try:
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(name=collection_name)
            app_logger.info(f"成功删除集合: {collection_name}")
            return True
        except Exception as e:
            app_logger.error(f"删除集合失败: {e}")
            return False


class QdrantVectorStore(VectorStoreBase):
    """Qdrant向量存储实现"""
    
    def __init__(self):
        try:
            # 尝试使用OpenAI embeddings（如果API密钥可用且有配额）
            if settings.openai_api_key:
                try:
                    # 测试OpenAI API是否可用
                    test_embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
                    # 进行一个小测试
                    test_embeddings.embed_query("test")
                    self.embeddings = test_embeddings
                    app_logger.info("Qdrant使用OpenAI Embeddings")
                except Exception as e:
                    app_logger.warning(f"OpenAI Embeddings不可用，Qdrant切换到本地模型: {e}")
                    # 使用本地HuggingFace模型
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'}
                    )
                    app_logger.info("Qdrant使用本地HuggingFace Embeddings")
            else:
                # 直接使用本地模型
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2", 
                    model_kwargs={'device': 'cpu'}
                )
                app_logger.info("Qdrant使用本地HuggingFace Embeddings")
        except Exception as e:
            app_logger.error(f"Qdrant Embedding模型初始化失败: {e}")
            raise RuntimeError("无法初始化任何embedding模型")
            
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key
        )
    
    async def add_documents(self, documents: List[Document], collection_name: str) -> List[str]:
        """添加文档到向量存储"""
        try:
            # 确保集合存在
            await self._ensure_collection(collection_name)
            
            vectorstore = Qdrant(
                client=self.client,
                collection_name=collection_name,
                embeddings=self.embeddings
            )
            
            ids = vectorstore.add_documents(documents)
            app_logger.info(f"成功添加 {len(documents)} 个文档到 {collection_name}")
            return ids
            
        except Exception as e:
            app_logger.error(f"添加文档失败: {e}")
            raise
    
    async def similarity_search(self, query: str, collection_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """相似度搜索"""
        try:
            vectorstore = Qdrant(
                client=self.client,
                collection_name=collection_name,
                embeddings=self.embeddings
            )
            
            results = vectorstore.similarity_search_with_score(query, k=top_k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            return formatted_results
            
        except Exception as e:
            app_logger.error(f"相似度搜索失败: {e}")
            raise
    
    async def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        try:
            self.client.delete_collection(collection_name=collection_name)
            app_logger.info(f"成功删除集合: {collection_name}")
            return True
        except Exception as e:
            app_logger.error(f"删除集合失败: {e}")
            return False
    
    async def _ensure_collection(self, collection_name: str):
        """确保集合存在"""
        collections = self.client.get_collections()
        if collection_name not in [col.name for col in collections.collections]:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
            )


class VectorService:
    """向量数据库服务"""
    
    def __init__(self, vector_store_type: str = "chroma"):
        self.vector_store_type = vector_store_type
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        if vector_store_type == "chroma":
            self.vector_store = ChromaVectorStore()
        elif vector_store_type == "qdrant":
            self.vector_store = QdrantVectorStore()
        else:
            raise ValueError(f"不支持的向量存储类型: {vector_store_type}")
    
    async def add_texts(
        self,
        texts: List[str],
        collection_name: str,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """添加文本文档"""
        try:
            documents = []
            for i, text in enumerate(texts):
                chunks = self.text_splitter.split_text(text)
                for chunk in chunks:
                    metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                    documents.append(Document(page_content=chunk, metadata=metadata))
            
            return await self.vector_store.add_documents(documents, collection_name)
            
        except Exception as e:
            app_logger.error(f"添加文本文档失败: {e}")
            raise
    
    async def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        try:
            results = await self.vector_store.similarity_search(query, collection_name, top_k)
            
            # 过滤低分结果
            filtered_results = [
                result for result in results 
                if result["score"] >= score_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            app_logger.error(f"搜索文档失败: {e}")
            raise
    
    async def delete_collection(self, collection_name: str) -> bool:
        """删除知识库"""
        return await self.vector_store.delete_collection(collection_name)


# 创建全局向量服务实例
vector_service = VectorService(vector_store_type="chroma") 