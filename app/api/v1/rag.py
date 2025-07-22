from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.response import FileUploadResponse
from app.core.rag_service import rag_service
from app.utils.logger import app_logger
import uuid

router = APIRouter()


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


@router.post("/rag/batch_documents")
async def batch_add_documents(
    collection_name: str,
    documents: List[dict]
):
    """批量添加文档到知识库"""
    try:
        result = await rag_service.batch_add_documents(
            documents=documents,
            collection_name=collection_name
        )
        
        return {
            "success": True,
            "message": result["message"],
            "collection_name": result["collection_name"],
            "added_count": result["added_count"]
        }
        
    except Exception as e:
        app_logger.error(f"批量添加文档失败: {e}")
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


@router.post("/rag/query")
async def query_knowledge_base(
    query: str,
    collection_name: str,
    top_k: int = 5
):
    """查询知识库"""
    try:
        result = await rag_service.query_knowledge_base(
            query=query,
            collection_name=collection_name,
            top_k=top_k
        )
        
        return {
            "success": True,
            "query": query,
            "collection_name": collection_name,
            "results": result.get("documents", []),
            "scores": result.get("scores", []),
            "total_found": len(result.get("documents", []))
        }
        
    except Exception as e:
        app_logger.error(f"知识库查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 