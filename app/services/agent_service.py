from typing import List, Dict, Any, Optional
import time
import asyncio
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.hub import pull
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.shell.tool import ShellTool
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from app.services.llm_service import llm_service
from app.services.memory_service import memory_service
from app.services.monitoring_service import monitoring_service
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


class EnhancedAgentService:
    """增强的Agent服务，集成记忆管理和ReAct Agent"""
    
    def __init__(self):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.monitoring_service = monitoring_service
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
            
            app_logger.info(f"成功初始化 {len(tools)} 个工具")
            return tools
            
        except Exception as e:
            app_logger.error(f"工具初始化失败: {e}")
            return []
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """获取可用工具列表"""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools
        ]
    
    async def execute_react_agent(
        self,
        task: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_iterations: int = 5,
        use_memory: bool = True
    ) -> Dict[str, Any]:
        """执行ReAct Agent任务"""
        start_time = time.time()
        trace_id = None
        steps = []
        
        try:
            # 获取LangFuse回调处理器
            langfuse_handler = self.monitoring_service.get_langchain_callback_handler()
            callbacks = [langfuse_handler] if langfuse_handler else []
            
            # 获取LLM模型
            if model not in self.llm_service._models:
                raise ValueError(f"模型 {model} 不可用")
            
            llm = self.llm_service._models[model]
            
            # 检索相关记忆
            relevant_memories = []
            if use_memory and self.memory_service.is_enabled():
                if user_id:
                    user_memories = await self.memory_service.search_user_memories(
                        query=task, user_id=user_id, limit=3
                    )
                    relevant_memories.extend(user_memories)
                
                if agent_id:
                    agent_memories = await self.memory_service.search_agent_memories(
                        query=task, agent_id=agent_id, limit=3
                    )
                    relevant_memories.extend(agent_memories)
                
                if session_id:
                    session_memories = await self.memory_service.search_session_memories(
                        query=task, session_id=session_id, limit=3
                    )
                    relevant_memories.extend(session_memories)
            
            # 构建带记忆的系统提示
            memory_context = ""
            if relevant_memories:
                memory_items = [f"- {mem.get('memory', '')}" for mem in relevant_memories[:5]]
                memory_context = f"\n\n相关记忆:\n" + "\n".join(memory_items)
            
            # 创建ReAct Agent
            try:
                # 使用LangChain Hub的ReAct prompt
                prompt = pull("hwchase17/react")
            except Exception:
                # 如果Hub不可用，使用自定义prompt
                template = """你是一个有用的AI助手，可以使用以下工具来帮助回答问题。

你有以下工具可用:
{tools}

工具使用格式:
```
Action: 工具名称
Action Input: 工具输入
```

请按照以下格式思考和行动:

Question: 用户的问题
Thought: 我需要思考如何解决这个问题
Action: 选择要使用的工具
Action Input: 工具的输入
Observation: 工具的输出
... (重复 Thought/Action/Action Input/Observation 直到有最终答案)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

{memory_context}

Question: {input}
Thought: {agent_scratchpad}"""
                
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["tools", "memory_context", "input", "agent_scratchpad"]
                )
            
            # 创建agent
            agent = create_react_agent(llm, self.tools, prompt)
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=max_iterations,
                handle_parsing_errors=True
            )
            
            # 执行任务 - 通过config传递callbacks和metadata
            result = await agent_executor.ainvoke(
                {
                    "input": task,
                    "memory_context": memory_context
                },
                config={
                    "callbacks": callbacks,
                    "metadata": {
                        "langfuse_user_id": user_id,
                        "langfuse_session_id": session_id,
                        "langfuse_tags": ["react-agent"],
                        "task": task,
                        "model": model,
                        "agent_id": agent_id,
                        "use_memory": use_memory
                    }
                }
            )
            
            execution_time = time.time() - start_time
            final_answer = result.get("output", "任务执行完成，但未获得明确结果")
            
            # 记录执行步骤
            steps = [
                {"step": 1, "action": "记忆检索", "result": f"找到 {len(relevant_memories)} 条相关记忆"},
                {"step": 2, "action": "Agent创建", "result": "成功创建ReAct Agent"},
                {"step": 3, "action": "任务执行", "result": "Agent开始执行任务"},
                {"step": 4, "action": "生成结果", "result": "任务执行完成"}
            ]
            
            # 保存新的记忆
            if use_memory and self.memory_service.is_enabled():
                messages = [
                    {"role": "user", "content": task},
                    {"role": "assistant", "content": final_answer}
                ]
                
                if user_id:
                    await self.memory_service.add_user_memory(
                        messages=messages,
                        user_id=user_id,
                        metadata={"task_type": "react_agent", "model": model}
                    )
                
                if agent_id:
                    await self.memory_service.add_agent_memory(
                        messages=messages,
                        agent_id=agent_id,
                        metadata={"task_type": "react_agent", "model": model}
                    )
                
                if session_id:
                    await self.memory_service.add_session_memory(
                        messages=messages,
                        session_id=session_id,
                        metadata={"task_type": "react_agent", "model": model}
                    )
            
# LangFuse会自动通过CallbackHandler记录执行详情
            tools_used = [tool.name for tool in self.tools]
            
            return {
                "success": True,
                "result": final_answer,
                "agent_type": "react_agent",
                "steps": steps,
                "tools_used": tools_used,
                "execution_time": execution_time,
                "memory_used": len(relevant_memories),
                "trace_id": langfuse_handler.last_trace_id if langfuse_handler else None
            }
            
        except Exception as e:
            app_logger.error(f"ReAct Agent执行失败: {e}")
            raise
        finally:
            # 刷新监控数据
            self.monitoring_service.flush()
    
    async def execute_simple_agent(
        self,
        task: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        model: str = "gpt-4o-mini",
        use_memory: bool = True
    ) -> Dict[str, Any]:
        """执行简单Agent任务（不使用工具）"""
        start_time = time.time()
        trace_id = None
        
        try:
            # 获取LangFuse回调处理器
            langfuse_handler = self.monitoring_service.get_langchain_callback_handler()
            callbacks = [langfuse_handler] if langfuse_handler else []
            
            # 检索相关记忆
            relevant_memories = []
            if use_memory and self.memory_service.is_enabled():
                if user_id:
                    user_memories = await self.memory_service.search_user_memories(
                        query=task, user_id=user_id, limit=5
                    )
                    relevant_memories.extend(user_memories)
            
            # 构建prompt
            memory_context = ""
            if relevant_memories:
                memory_items = [f"- {mem.get('memory', '')}" for mem in relevant_memories]
                memory_context = f"\n\n基于以下相关记忆来回答:\n" + "\n".join(memory_items)
            
            # 构建消息
            messages = []
            if memory_context:
                messages.append({"role": "system", "content": f"你是一个智能助手。基于以下相关记忆来回答：{memory_context}"})
            else:
                messages.append({"role": "system", "content": "你是一个智能助手。"})
            messages.append({"role": "user", "content": task})
            
            # 获取LLM模型并调用
            if model not in self.llm_service._models:
                raise ValueError(f"模型 {model} 不可用")
            
            llm = self.llm_service._models[model]
            
            # 通过LangChain调用，启用LangFuse追踪
            from langchain_core.messages import HumanMessage, SystemMessage
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                else:
                    langchain_messages.append(HumanMessage(content=msg["content"]))
            
            response = await llm.ainvoke(
                langchain_messages,
                config={
                    "callbacks": callbacks,
                    "metadata": {
                        "langfuse_user_id": user_id,
                        "langfuse_session_id": session_id,
                        "langfuse_tags": ["simple-agent"],
                        "task": task,
                        "model": model
                    }
                }
            )
            
            execution_time = time.time() - start_time
            final_answer = response.content
            
            # 记录执行步骤
            steps = [
                {"step": 1, "action": "记忆检索", "result": f"找到 {len(relevant_memories)} 条相关记忆"},
                {"step": 2, "action": "LLM调用", "result": "生成回答"},
                {"step": 3, "action": "任务完成", "result": "返回结果"}
            ]
            
            # 保存新的记忆
            if use_memory and self.memory_service.is_enabled():
                messages = [
                    {"role": "user", "content": task},
                    {"role": "assistant", "content": final_answer}
                ]
                
                if user_id:
                    await self.memory_service.add_user_memory(
                        messages=messages,
                        user_id=user_id,
                        metadata={"task_type": "simple_agent", "model": model}
                    )
            
            return {
                "success": True,
                "result": final_answer,
                "agent_type": "simple_agent",
                "steps": steps,
                "tools_used": [],
                "execution_time": execution_time,
                "memory_used": len(relevant_memories),
                "trace_id": langfuse_handler.last_trace_id if langfuse_handler else None
            }
            
        except Exception as e:
            app_logger.error(f"简单Agent执行失败: {e}")
            raise
        finally:
            # 刷新监控数据
            self.monitoring_service.flush()
    
    async def get_memory_summary(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取记忆摘要"""
        if not self.memory_service.is_enabled():
            return {"enabled": False, "message": "记忆服务未启用"}
        
        summary = {"enabled": True}
        
        if user_id:
            user_memories = await self.memory_service.get_all_user_memories(user_id, limit=5)
            summary["user_memories"] = len(user_memories)
            summary["recent_user_memories"] = user_memories[:3]
        
        if agent_id:
            agent_memories = await self.memory_service.get_all_agent_memories(agent_id, limit=5)
            summary["agent_memories"] = len(agent_memories)
            summary["recent_agent_memories"] = agent_memories[:3]
        
        return summary


# 创建全局增强Agent服务实例
enhanced_agent_service = EnhancedAgentService() 