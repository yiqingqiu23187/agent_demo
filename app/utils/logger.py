import sys
from loguru import logger
from app.core.config import settings


def setup_logger():
    """配置Loguru日志器"""
    # 移除默认的日志器
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # 添加文件输出
    logger.add(
        "logs/app.log",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="10 days",
        compression="zip"
    )
    
    return logger


# 创建全局日志器实例
app_logger = setup_logger() 