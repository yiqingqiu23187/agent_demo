import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import settings
from app.utils.logger import app_logger
from app.api.v1 import rag, agents
from app.models.response import HealthResponse, ErrorResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    app_logger.info("🚀 应用启动中...")
    
    # 创建必要的目录
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    app_logger.info("✅ 应用启动完成")
    
    yield
    
    # 关闭时执行
    app_logger.info("🔄 应用关闭中...")
    
    # 这里可以添加清理逻辑
    from app.core.monitoring_service import monitoring_service
    monitoring_service.flush()
    
    app_logger.info("✅ 应用已关闭")


# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI Agent应用开发演示项目",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加可信主机中间件
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", settings.host]
    )


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    app_logger.error(f"未处理的异常: {exc}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc) if settings.debug else "服务器内部错误"
        ).model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            detail=exc.detail
        ).model_dump()
    )


# 健康检查端点
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    try:
        # 检查各个服务状态
        from app.core.llm_service import llm_service
        from app.core.vector_service import vector_service
        from app.core.monitoring_service import monitoring_service
        from app.core.memory_service import memory_service
        
        services = {
            "llm_service": "healthy" if llm_service.get_available_models() else "unhealthy",
            "vector_service": "healthy" if vector_service else "unhealthy",
            "monitoring_service": "healthy" if monitoring_service.is_enabled() else "disabled",
            "memory_service": "healthy" if memory_service.is_enabled() else "disabled"
        }
        
        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            services=services
        )
        
    except Exception as e:
        app_logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="健康检查失败")


# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": f"欢迎使用 {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


# 包含API路由
from app.api.v1 import rag, agents

app.include_router(rag.router, prefix="/api/v1", tags=["知识库管理"])
app.include_router(agents.router, prefix="/api/v1", tags=["智能体"])


def main():
    """启动应用"""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main() 