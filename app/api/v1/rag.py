from typing import Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from app.models.request import RAGQueryRequest, DocumentUploadRequest
from app.models.response import RAGResponse, FileUploadResponse
from app.services.rag_service import rag_service
from app.utils.logger import app_logger
import uuid

router = APIRouter()


@router.post("/rag/query", response_model=RAGResponse)
async def rag_query(
    request: RAGQueryRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
):
    """RAG查询接口"""
    try:
        result = await rag_service.query(
            query=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            user_id=user_id,
            session_id=session_id
        )
        
        return RAGResponse(
            success=True,
            message="RAG查询完成",
            answer=result["answer"],
            sources=result["sources"],
            query=request.query,
            collection_name=request.collection_name
        )
        
    except Exception as e:
        app_logger.error(f"RAG查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/documents", response_model=FileUploadResponse)
async def add_documents(
    collection_name: str,
    texts: List[str],
    metadatas: Optional[List[dict]] = None
):
    """添加文档到知识库"""
    try:
        result = await rag_service.add_documents_to_knowledge_base(
            texts=texts,
            collection_name=collection_name,
            metadatas=metadatas
        )
        
        return FileUploadResponse(
            success=True,
            message=result["message"],
            file_id=str(uuid.uuid4()),
            file_name="text_documents",
            collection_name=collection_name,
            chunks_count=len(texts),
            status="completed"
        )
        
    except Exception as e:
        app_logger.error(f"添加文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/upload")
async def upload_file(
    collection_name: str,
    file: UploadFile = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """上传文件到知识库"""
    try:
        # 读取文件内容
        content = await file.read()
        
        # 根据文件类型处理文本
        if file.content_type == "text/plain":
            text = content.decode("utf-8")
        elif file.content_type == "application/pdf":
            # 这里可以添加PDF处理逻辑
            text = "PDF文件处理需要额外的依赖包"
        else:
            raise HTTPException(status_code=400, detail="不支持的文件类型")
        
        # 添加到知识库
        result = await rag_service.add_documents_to_knowledge_base(
            texts=[text],
            collection_name=collection_name,
            metadatas=[{"filename": file.filename, "content_type": file.content_type}]
        )
        
        return {
            "success": True,
            "message": f"文件 {file.filename} 上传成功",
            "file_name": file.filename,
            "collection_name": collection_name,
            "file_size": len(content)
        }
        
    except Exception as e:
        app_logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/collections/{collection_name}")
async def get_collection_info(collection_name: str):
    """获取知识库信息"""
    try:
        info = await rag_service.get_collection_info(collection_name)
        return {
            "success": True,
            "collection_info": info
        }
    except Exception as e:
        app_logger.error(f"获取知识库信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rag/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """删除知识库"""
    try:
        result = await rag_service.delete_collection(collection_name)
        return result
    except Exception as e:
        app_logger.error(f"删除知识库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 