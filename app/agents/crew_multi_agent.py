from typing import List, Dict, Any, Optional
import time
import asyncio
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.hub import pull
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.llm_service import llm_service
from app.core.memory_service import memory_service  
from app.core.monitoring_service import monitoring_service
from app.core.tool_service import tool_service
from app.utils.logger import app_logger


class LangChainReactCrewAgent(Agent):
    """
    LangChain ReAct Agent的CrewAI适配器
    继承CrewAI的Agent类，但底层使用LangChain的ReAct agent实现
    """
    
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: List[BaseTool],
        llm,
        memory_context: str = "",
        max_iterations: int = 5,
        **kwargs
    ):
        # 初始化CrewAI Agent的基本属性
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools,
            llm=llm,
            verbose=kwargs.get('verbose', True),
            allow_delegation=kwargs.get('allow_delegation', False),
            **kwargs
        )
        
        self.memory_context = memory_context
        self.max_iterations = max_iterations
        self.langchain_agent_executor = None
        
        # 初始化LangChain ReAct agent
        self._initialize_langchain_agent()
    
    def _initialize_langchain_agent(self):
        """初始化LangChain ReAct agent"""
        try:
            # 创建自定义prompt，包含CrewAI的角色信息
            prompt = self._create_crew_aware_prompt()
            
            # 创建LangChain ReAct agent
            langchain_agent = create_react_agent(self.llm, self.tools, prompt)
            
            # 创建AgentExecutor
            self.langchain_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=langchain_agent,
                tools=self.tools,
                verbose=self.verbose,
                max_iterations=self.max_iterations,
                handle_parsing_errors=True
            )
            
            app_logger.info(f"LangChain ReAct agent initialized for role: {self.role}")
            
        except Exception as e:
            app_logger.error(f"Failed to initialize LangChain agent: {e}")
            # 如果初始化失败，回退到父类的默认行为
            self.langchain_agent_executor = None
    
    def _create_crew_aware_prompt(self) -> PromptTemplate:
        """创建包含CrewAI角色信息的prompt"""
        try:
            # 尝试使用LangChain Hub的ReAct prompt作为基础
            base_prompt = pull("hwchase17/react")
            template = base_prompt.template
        except Exception:
            # 如果Hub不可用，使用自定义的基础template
            template = """You are a helpful AI assistant that can use tools to help answer questions.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}"""
        
        # 增强prompt，包含CrewAI的角色信息
        enhanced_template = f"""You are playing the role of: {self.role}

Your goal is: {self.goal}

Your background: {self.backstory}

{self.memory_context}

{template}"""
        
        return PromptTemplate(
            template=enhanced_template,
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
        )
    
    async def execute_task(self, task_description: str) -> str:
        """
        执行任务的主要方法
        这个方法会被CrewAI框架调用
        """
        try:
            if self.langchain_agent_executor is None:
                # 如果LangChain agent初始化失败，使用父类的默认实现
                app_logger.warning(f"LangChain agent not available for {self.role}, using default CrewAI agent")
                return await super().execute_task(task_description)
            
            # 使用LangChain ReAct agent执行任务
            app_logger.info(f"{self.role} 开始执行任务: {task_description}")
            
            # 异步执行LangChain agent
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.langchain_agent_executor.invoke({
                    "input": task_description
                })
            )
            
            final_answer = result.get("output", "任务执行完成，但未获得明确结果")
            app_logger.info(f"{self.role} 完成任务，结果: {final_answer[:100]}...")
            
            return final_answer
            
        except Exception as e:
            app_logger.error(f"{self.role} 执行任务失败: {e}")
            return f"任务执行失败: {str(e)}"


class CrewMultiAgent:
    """CrewAI 多智能体系统 - 使用LangChain ReAct Agent作为底层实现"""
    
    def __init__(self):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.monitoring_service = monitoring_service
        self.tool_service = tool_service
        self.agent_type = "crew"
        
        # 预定义的智能体角色
        self.predefined_roles = {
            "researcher": {
                "role": "Senior Research Analyst",
                "goal": "收集、分析和整理相关信息，为团队提供准确的研究支持",
                "backstory": "你是一位经验丰富的研究分析师，擅长从各种来源收集信息，具有敏锐的洞察力和严谨的研究方法。你能够快速识别关键信息，并进行深入的背景调研。"
            },
            "analyst": {
                "role": "Data Analyst", 
                "goal": "分析数据和信息，提取关键见解，为决策提供支持",
                "backstory": "你是一位专业的数据分析师，能够从复杂的数据中发现模式和趋势，提供有价值的分析结果。你具有强大的逻辑思维和分析能力。"
            },
            "writer": {
                "role": "Content Writer",
                "goal": "将分析结果和信息整理成清晰、有条理的内容",
                "backstory": "你是一位专业的内容创作者，擅长将复杂的信息转化为易懂、引人入胜的内容。你注重逻辑结构和表达清晰度。"
            },
            "reviewer": {
                "role": "Quality Reviewer",
                "goal": "审查和优化内容质量，确保准确性和完整性",
                "backstory": "你是一位严谨的质量审查员，具有敏锐的眼光和高标准，能够发现问题并提出改进建议。你对细节要求极高。"
            },
            "coordinator": {
                "role": "Project Coordinator",
                "goal": "协调团队工作，确保任务顺利完成和信息流通",
                "backstory": "你是一位经验丰富的项目协调员，擅长统筹规划和团队协作，能够确保项目按时高质量完成。"
            }
        }
    
    async def execute(
        self,
        task: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        model: str = "gpt-4o-mini",
        use_memory: bool = True,
        selected_tools: Optional[List[str]] = None,
        crew_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行多智能体协作任务"""
        start_time = time.time()
        
        try:
            app_logger.info(f"开始执行Crew多智能体任务: {task}")
            
            # 获取LangFuse回调处理器（用于监控）
            langfuse_handler = self.monitoring_service.get_langchain_callback_handler()
            
            # 获取工具
            tools = await self._get_tools(selected_tools)
            
            # 检索相关记忆
            relevant_memories = await self._retrieve_memories(
                task, user_id, agent_id, session_id, use_memory
            )
            
            # 构建记忆上下文
            memory_context = self._build_memory_context(relevant_memories)
            
            # 根据任务复杂度选择合适的crew配置
            if crew_config:
                agents_config = crew_config.get("agents", ["researcher", "analyst", "writer"])
                process_type = crew_config.get("process", "sequential")
                max_iterations = crew_config.get("max_iterations", 5)
            else:
                agents_config, process_type = self._auto_select_crew_config(task)
                max_iterations = 5
            
            # 获取LLM模型
            if model not in self.llm_service._models:
                raise ValueError(f"模型 {model} 不可用")
            llm = self.llm_service._models[model]
            
            # 创建LangChain-based智能体团队
            agents = await self._create_langchain_agents(
                agents_config, llm, tools, memory_context, max_iterations
            )
            
            # 创建任务分解
            tasks = await self._create_tasks(task, agents, agents_config)
            
            # 创建并执行Crew
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential if process_type == "sequential" else Process.hierarchical,
                verbose=True
            )
            
            # 执行多智能体协作
            app_logger.info(f"启动{len(agents)}个LangChain-based智能体协作执行任务")
            result = await asyncio.get_event_loop().run_in_executor(
                None, crew.kickoff
            )
            
            execution_time = time.time() - start_time
            
            # 保存协作记忆
            await self._save_memories(
                task, str(result), user_id, agent_id, session_id, 
                use_memory, model
            )
            
            tools_used = [tool.name for tool in tools] if tools else []
            
            app_logger.info(f"Crew多智能体任务执行完成，耗时: {execution_time:.2f}秒")
            
            return {
                "success": True,
                "result": str(result),
                "agent_type": self.agent_type,
                "tools_used": tools_used,
                "execution_time": execution_time,
                "memory_used": len(relevant_memories),
                "agents_used": agents_config,
                "process_type": process_type,
                "agents_count": len(agents)
            }
            
        except Exception as e:
            app_logger.error(f"Crew多智能体执行失败: {e}")
            return {
                "success": False,
                "result": f"Crew多智能体执行失败: {str(e)}",
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
            tools = []
            for tool_name in selected_tools:
                try:
                    tool = self.tool_service.get_tool_by_name(tool_name)
                    tools.append(tool)
                except ValueError:
                    app_logger.warning(f"工具 {tool_name} 未找到，跳过")
            return tools
        else:
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
        return f"\n\n相关历史记忆:\n" + "\n".join(memory_items)
    
    def _auto_select_crew_config(self, task: str) -> tuple:
        """根据任务自动选择crew配置"""
        task_lower = task.lower()
        
        # 分析类任务
        if any(keyword in task_lower for keyword in ["分析", "研究", "调研", "分析报告"]):
            return ["researcher", "analyst", "writer"], "sequential"
        
        # 创作类任务
        elif any(keyword in task_lower for keyword in ["写", "创作", "文章", "内容"]):
            return ["researcher", "writer", "reviewer"], "sequential"
        
        # 复杂决策类任务
        elif any(keyword in task_lower for keyword in ["决策", "方案", "计划", "策略"]):
            return ["researcher", "analyst", "writer", "reviewer"], "sequential"
        
        # 协调类任务
        elif any(keyword in task_lower for keyword in ["协调", "管理", "整合"]):
            return ["coordinator", "researcher", "writer"], "sequential"
        
        # 默认配置
        else:
            return ["researcher", "writer"], "sequential"
    
    async def _create_langchain_agents(
        self, 
        agents_config: List[str], 
        llm,
        tools: List[BaseTool],
        memory_context: str,
        max_iterations: int
    ) -> List[LangChainReactCrewAgent]:
        """创建基于LangChain的智能体团队"""
        agents = []
        
        for agent_key in agents_config:
            if agent_key not in self.predefined_roles:
                app_logger.warning(f"未知的智能体角色: {agent_key}")
                continue
            
            role_config = self.predefined_roles[agent_key]
            
            # 创建LangChain-based CrewAI Agent
            agent = LangChainReactCrewAgent(
                role=role_config["role"],
                goal=role_config["goal"],
                backstory=role_config["backstory"],
                tools=tools,
                llm=llm,
                memory_context=memory_context,
                max_iterations=max_iterations,
                verbose=True,
                allow_delegation=False
            )
            
            agents.append(agent)
            app_logger.info(f"创建LangChain-based智能体: {role_config['role']}")
        
        return agents
    
    async def _create_tasks(
        self, 
        main_task: str, 
        agents: List[LangChainReactCrewAgent], 
        agents_config: List[str]
    ) -> List[Task]:
        """创建任务分解"""
        tasks = []
        
        # 根据智能体配置创建相应的任务
        for i, (agent, agent_key) in enumerate(zip(agents, agents_config)):
            if agent_key == "researcher":
                task = Task(
                    description=f"针对以下任务进行深入研究和信息收集：{main_task}。收集相关信息、数据和背景资料，使用可用的工具进行搜索和分析。",
                    expected_output="详细的研究报告，包含收集到的关键信息和数据",
                    agent=agent
                )
            elif agent_key == "analyst":
                task = Task(
                    description=f"基于前面的研究结果，对以下任务进行深入分析：{main_task}。提取关键见解和模式，进行数据分析。",
                    expected_output="分析报告，包含关键见解、趋势分析和结论",
                    agent=agent
                )
            elif agent_key == "writer":
                task = Task(
                    description=f"将研究和分析结果整理成清晰的内容：{main_task}。确保内容结构清晰、逻辑性强，易于理解。",
                    expected_output="结构化的内容，包含清晰的结论和建议",
                    agent=agent
                )
            elif agent_key == "reviewer":
                task = Task(
                    description=f"审查和优化前面生成的内容：{main_task}。检查准确性、完整性和质量，提出改进建议。",
                    expected_output="经过优化的最终内容，确保高质量和准确性",
                    agent=agent
                )
            elif agent_key == "coordinator":
                task = Task(
                    description=f"协调和整合团队工作成果：{main_task}。确保各部分工作协调一致，信息流通顺畅。",
                    expected_output="协调后的整合方案和执行建议",
                    agent=agent
                )
            else:
                # 通用任务
                task = Task(
                    description=f"处理以下任务：{main_task}",
                    expected_output="任务处理结果",
                    agent=agent
                )
            
            tasks.append(task)
            app_logger.info(f"创建任务 {i+1}: {agent.role}")
        
        return tasks
    
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
        """保存协作记忆"""
        if not use_memory or not self.memory_service.is_enabled():
            return
        
        try:
            messages = [
                {"role": "user", "content": task},
                {"role": "assistant", "content": result}
            ]
            
            metadata = {"task_type": "crew_multi_agent", "model": model}
            
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
            "name": "CrewAI Multi-Agent with LangChain",
            "description": "基于CrewAI的多智能体协作系统，底层使用LangChain ReAct Agent实现",
            "supports_tools": True,
            "supports_memory": True,
            "available_models": self.llm_service.get_available_models(),
            "available_tools": self.tool_service.get_tools_info(),
            "predefined_roles": list(self.predefined_roles.keys()),
            "supported_processes": ["sequential", "hierarchical"],
            "framework": "CrewAI + LangChain"
        }
    
    def get_predefined_roles(self) -> Dict[str, Dict[str, str]]:
        """获取预定义角色信息"""
        return self.predefined_roles


# 创建全局CrewAI多智能体实例
crew_multi_agent = CrewMultiAgent() 