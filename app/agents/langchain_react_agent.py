from typing import List, Dict, Any, Optional
import time
import asyncio
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.hub import pull
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from app.core.llm_service import llm_service
from app.core.memory_service import memory_service  
from app.core.monitoring_service import monitoring_service
from app.core.tool_service import tool_service
from app.utils.logger import app_logger


class LangChainReactAgent:
    """LangChain ReAct Agent - 基于推理-行动循环的智能体"""
    
    def __init__(self):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.monitoring_service = monitoring_service
        self.tool_service = tool_service
        self.agent_type = "react"
    
    async def execute(
        self,
        task: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_iterations: int = 5,
        use_memory: bool = True,
        selected_tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """执行ReAct Agent任务"""
        start_time = time.time()
        
        try:
            app_logger.info(f"开始执行ReAct Agent任务: {task}")
            
            # 获取LangFuse回调处理器
            langfuse_handler = self.monitoring_service.get_langchain_callback_handler()
            callbacks = [langfuse_handler] if langfuse_handler else []
            
            # 获取LLM模型
            if model not in self.llm_service._models:
                raise ValueError(f"模型 {model} 不可用")
            
            llm = self.llm_service._models[model]
            
            # 获取工具
            tools = await self._get_tools(selected_tools)
            if not tools:
                raise ValueError("没有可用的工具来执行ReAct Agent")
            
            # 检索相关记忆
            relevant_memories = await self._retrieve_memories(
                task, user_id, agent_id, session_id, use_memory
            )
            
            # 构建带记忆的系统提示
            memory_context = self._build_memory_context(relevant_memories)
            
            # 创建ReAct Agent
            prompt = await self._create_prompt(memory_context)
            agent = create_react_agent(llm, tools, prompt)
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=max_iterations,
                handle_parsing_errors=True
            )
            
            # 执行任务
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
                        "use_memory": use_memory,
                        "selected_tools": selected_tools
                    }
                }
            )
            
            execution_time = time.time() - start_time
            final_answer = result.get("output", "任务执行完成，但未获得明确结果")
            
            # 保存新的记忆
            await self._save_memories(
                task, final_answer, user_id, agent_id, session_id, 
                use_memory, model
            )
            
            tools_used = [tool.name for tool in tools]
            
            app_logger.info(f"ReAct Agent任务执行完成，耗时: {execution_time:.2f}秒")
            
            return {
                "success": True,
                "result": final_answer,
                "agent_type": self.agent_type,
                "tools_used": tools_used,
                "execution_time": execution_time,
                "memory_used": len(relevant_memories),
                "trace_id": langfuse_handler.last_trace_id if langfuse_handler else None
            }
            
        except Exception as e:
            app_logger.error(f"ReAct Agent执行失败: {e}")
            return {
                "success": False,
                "result": f"ReAct Agent执行失败: {str(e)}",
                "agent_type": self.agent_type,
                "tools_used": [],
                "execution_time": time.time() - start_time,
                "memory_used": 0,
                "error": str(e)
            }
        finally:
            # 刷新监控数据
            self.monitoring_service.flush()
    
    async def _get_tools(self, selected_tools: Optional[List[str]] = None) -> List[BaseTool]:
        """获取工具列表"""
        if selected_tools:
            # 如果指定了特定工具，只使用这些工具
            tools = []
            for tool_name in selected_tools:
                try:
                    tool = self.tool_service.get_tool_by_name(tool_name)
                    tools.append(tool)
                except ValueError:
                    app_logger.warning(f"工具 {tool_name} 未找到，跳过")
            return tools
        else:
            # 使用所有可用工具
            return self.tool_service.get_available_tools()
    
    async def _retrieve_memories(
        self, 
        task: str, 
        user_id: Optional[str], 
        agent_id: Optional[str], 
        session_id: Optional[str], 
        use_memory: bool
    ) -> List[Dict[str, Any]]:
        """检索相关记忆"""
        relevant_memories = []
        
        if not use_memory or not self.memory_service.is_enabled():
            return relevant_memories
        
        try:
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
                
        except Exception as e:
            app_logger.warning(f"记忆检索失败: {e}")
        
        return relevant_memories
    
    def _build_memory_context(self, relevant_memories: List[Dict[str, Any]]) -> str:
        """构建记忆上下文"""
        if not relevant_memories:
            return ""
        
        memory_items = [f"- {mem.get('memory', '')}" for mem in relevant_memories[:5]]
        return f"\n\n相关记忆:\n" + "\n".join(memory_items)
    
    async def _create_prompt(self, memory_context: str) -> PromptTemplate:
        """创建prompt模板"""
        try:
            # 尝试使用LangChain Hub的ReAct prompt
            prompt = pull("hwchase17/react")
            return prompt
        except Exception as e:
            app_logger.warning(f"无法从LangChain Hub获取prompt，使用自定义prompt: {e}")
            
            # 使用自定义prompt
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
            
            return PromptTemplate(
                template=template,
                input_variables=["tools", "memory_context", "input", "agent_scratchpad"]
            )
    
    async def _save_memories(
        self, 
        task: str, 
        result: str, 
        user_id: Optional[str], 
        agent_id: Optional[str], 
        session_id: Optional[str],
        use_memory: bool,
        model: str
    ):
        """保存新的记忆"""
        if not use_memory or not self.memory_service.is_enabled():
            return
        
        try:
            messages = [
                {"role": "user", "content": task},
                {"role": "assistant", "content": result}
            ]
            
            metadata = {"task_type": "react_agent", "model": model}
            
            if user_id:
                await self.memory_service.add_user_memory(
                    messages=messages,
                    user_id=user_id,
                    metadata=metadata
                )
            
            if agent_id:
                await self.memory_service.add_agent_memory(
                    messages=messages,
                    agent_id=agent_id,
                    metadata=metadata
                )
            
            if session_id:
                await self.memory_service.add_session_memory(
                    messages=messages,
                    session_id=session_id,
                    metadata=metadata
                )
        except Exception as e:
            app_logger.warning(f"保存记忆失败: {e}")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """获取Agent信息"""
        return {
            "agent_type": self.agent_type,
            "name": "LangChain ReAct Agent",
            "description": "基于推理-行动循环的智能体，能够使用工具解决复杂问题",
            "supports_tools": True,
            "supports_memory": True,
            "available_models": self.llm_service.get_available_models(),
            "available_tools": self.tool_service.get_tools_info()
        }


# 创建全局LangChain ReAct Agent实例
langchain_react_agent = LangChainReactAgent() 