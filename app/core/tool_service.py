from typing import List, Dict, Any
from langchain.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.shell.tool import ShellTool
from app.utils.logger import app_logger


class CalculatorTool(BaseTool):
    """计算器工具"""
    name: str = "calculator"
    description: str = "用于计算数学表达式。输入应该是一个有效的数学表达式。"
    
    def _run(self, expression: str) -> str:
        try:
            # 安全的数学计算
            result = eval(expression, {"__builtins__": {}}, {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow, "sqrt": lambda x: x**0.5
            })
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        return self._run(expression)


class WeatherTool(BaseTool):
    """天气查询工具（模拟）"""
    name: str = "weather"
    description: str = "查询指定城市的天气信息。输入应该是城市名称。"
    
    def _run(self, city: str) -> str:
        # 这里是模拟的天气数据，实际应该调用天气API
        weather_data = {
            "北京": "晴天，温度 15°C，湿度 45%",
            "上海": "多云，温度 18°C，湿度 65%", 
            "深圳": "雨天，温度 22°C，湿度 80%",
            "广州": "晴天，温度 25°C，湿度 55%"
        }
        
        result = weather_data.get(city, f"抱歉，暂未找到 {city} 的天气信息")
        return f"{city} 的天气: {result}"
    
    async def _arun(self, city: str) -> str:
        return self._run(city)


class ToolService:
    """工具调用服务 - 管理所有可用工具"""
    
    def __init__(self):
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self) -> List[BaseTool]:
        """初始化工具集"""
        tools = []
        
        try:
            # 基础工具
            tools.extend([
                CalculatorTool(),
                WeatherTool(),
            ])
            
            # 搜索工具
            try:
                search_tool = DuckDuckGoSearchRun()
                search_tool.name = "search"
                search_tool.description = "用于搜索网络信息。输入应该是搜索查询。"
                tools.append(search_tool)
            except Exception as e:
                app_logger.warning(f"搜索工具初始化失败: {e}")
            
            # Shell工具（可选，根据需要启用）
            # try:
            #     shell_tool = ShellTool()
            #     tools.append(shell_tool)
            # except Exception as e:
            #     app_logger.warning(f"Shell工具初始化失败: {e}")
            
            app_logger.info(f"成功初始化 {len(tools)} 个工具")
            return tools
            
        except Exception as e:
            app_logger.error(f"工具初始化失败: {e}")
            return []
    
    def get_available_tools(self) -> List[BaseTool]:
        """获取所有可用工具实例"""
        return self.tools
    
    def get_tools_info(self) -> List[Dict[str, str]]:
        """获取工具信息列表"""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools
        ]
    
    def get_tool_by_name(self, tool_name: str) -> BaseTool:
        """根据名称获取特定工具"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"工具 {tool_name} 未找到")
    
    async def execute_tool(self, tool_name: str, tool_input: str) -> str:
        """执行指定工具"""
        try:
            tool = self.get_tool_by_name(tool_name)
            result = await tool._arun(tool_input)
            return result
        except Exception as e:
            app_logger.error(f"执行工具 {tool_name} 失败: {e}")
            return f"工具执行失败: {str(e)}"
    
    def add_custom_tool(self, tool: BaseTool):
        """添加自定义工具"""
        if tool not in self.tools:
            self.tools.append(tool)
            app_logger.info(f"添加自定义工具: {tool.name}")
    
    def remove_tool(self, tool_name: str):
        """移除工具"""
        self.tools = [tool for tool in self.tools if tool.name != tool_name]
        app_logger.info(f"移除工具: {tool_name}")


# 创建全局工具服务实例
tool_service = ToolService() 