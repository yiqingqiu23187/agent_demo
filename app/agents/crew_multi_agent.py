from typing import List, Dict, Any, Optional
import time
import asyncio
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from app.core.llm_service import llm_service
from app.core.memory_service import memory_service  
from app.core.monitoring_service import monitoring_service
from app.core.tool_service import tool_service
from app.utils.logger import app_logger


# 自定义工具演示
@tool("search_local_knowledge")
def search_local_knowledge(query: str) -> str:
    """搜索本地知识库，查找相关信息"""
    try:
        # 这里可以集成现有的向量搜索服务
        app_logger.info(f"搜索本地知识库: {query}")
        return f"本地知识库搜索结果: 找到与'{query}'相关的信息"
    except Exception as e:
        return f"本地知识库搜索失败: {str(e)}"


@tool("analyze_data")
def analyze_data(data_description: str) -> str:
    """分析提供的数据并生成见解"""
    try:
        app_logger.info(f"分析数据: {data_description}")
        return f"数据分析结果: 基于'{data_description}'的分析显示了关键趋势和模式"
    except Exception as e:
        return f"数据分析失败: {str(e)}"


class CrewMultiAgent:
    """CrewAI 原生多智能体系统 - 演示RAG、Memory、Tools功能"""
    
    def __init__(self):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.monitoring_service = monitoring_service
        self.tool_service = tool_service
        self.agent_type = "crewai_native"
        
        # 初始化CrewAI工具
        self._initialize_tools()
        
        # 预定义的智能体角色配置
        self.agent_configs = {
            "researcher": {
                "role": "高级研究分析师",
                "goal": "收集、分析和整理相关信息，为团队提供准确的研究支持",
                "backstory": "你是一位经验丰富的研究分析师，擅长从各种来源收集信息，具有敏锐的洞察力和严谨的研究方法。你精通使用各种搜索和分析工具来获取准确的信息。",
                "tools": ["search_tools", "rag_tools", "web_tools"]
            },
            "analyst": {
                "role": "数据分析专家", 
                "goal": "分析数据和信息，提取关键见解，为决策提供支持",
                "backstory": "你是一位专业的数据分析师，能够从复杂的数据中发现模式和趋势，提供有价值的分析结果。你善于使用各种分析工具来处理和理解数据。",
                "tools": ["analysis_tools", "rag_tools", "data_tools"]
            },
            "writer": {
                "role": "内容创作专家",
                "goal": "将分析结果和信息整理成清晰、有条理的内容",
                "backstory": "你是一位专业的内容创作者，擅长将复杂的信息转化为易懂、引人入胜的内容。你注重逻辑结构和表达清晰度，能够创作各种类型的文档。",
                "tools": ["document_tools", "rag_tools", "web_tools"]
            },
            "coordinator": {
                "role": "项目协调员",
                "goal": "协调团队工作，确保任务顺利完成和信息流通",
                "backstory": "你是一位经验丰富的项目协调员，擅长统筹规划和团队协作，能够确保项目按时高质量完成。你善于整合不同来源的信息并做出协调决策。",
                "tools": ["coordination_tools", "rag_tools", "analysis_tools"]
            },
            "reviewer": {
                "role": "质量审查员",
                "goal": "审查和优化内容质量，确保准确性和完整性",
                "backstory": "你是一位严谨的质量审查员，具有敏锐的眼光和高标准，能够发现问题并提出改进建议。你对细节要求极高，确保最终输出的质量。",
                "tools": ["review_tools", "document_tools", "rag_tools"]
            }
        }
    
    def _initialize_tools(self):
        """初始化CrewAI工具集合 - 只包含稳定可靠的工具"""
        try:
            # 基础RAG工具集合（稳定可靠）
            self.rag_tools = [
                search_local_knowledge,  # 自定义本地知识搜索
            ]
            # 数据分析工具集合（自定义工具，稳定可靠）
            self.analysis_tools = [
                analyze_data,  # 自定义数据分析工具
            ]
            # 文档工具集合（基础工具）
            self.document_tools = []
            # 整合所有工具
            self.all_tools = {
                "rag_tools": self.rag_tools,
                "analysis_tools": self.analysis_tools,
                "data_tools": self.analysis_tools,  # 别名
                "document_tools": self.document_tools,
                "coordination_tools": self.rag_tools + self.analysis_tools,
                "review_tools": self.document_tools + self.rag_tools,
            }
            app_logger.info("CrewAI工具初始化完成")
        except Exception as e:
            app_logger.error(f"工具初始化失败: {e}")
            # 设置基础工具作为后备
            self.rag_tools = []
            self.all_tools = {"rag_tools": self.rag_tools}
    
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
        """执行CrewAI原生多智能体协作任务"""
        start_time = time.time()
        
        try:
            app_logger.info(f"开始执行CrewAI原生多智能体任务: {task}")
            
            # 获取LLM模型
            if model not in self.llm_service._models:
                raise ValueError(f"模型 {model} 不可用")
            llm = self.llm_service._models[model]
            
            # 根据任务复杂度选择合适的crew配置
            if crew_config:
                agents_config = crew_config.get("agents", ["researcher", "writer"])
                process_type = crew_config.get("process", "sequential")
                max_iterations = crew_config.get("max_iterations", 5)
            else:
                agents_config, process_type = self._auto_select_crew_config(task)
                max_iterations = 5
            
            # 创建CrewAI智能体团队
            agents = self._create_crewai_agents(agents_config, llm)
            
            # 创建任务分解
            tasks = self._create_crewai_tasks(task, agents, agents_config)
            
            # 配置记忆系统
            memory_config = self._configure_memory(use_memory, user_id, session_id)
            
            # 创建并执行Crew
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential if process_type == "sequential" else Process.hierarchical,
                memory=use_memory,  # 启用CrewAI原生记忆系统
                cache=True,  # 启用缓存
                max_rpm=100,  # 限制每分钟请求数
                verbose=True,
                # 嵌入模型配置（使用OpenAI作为默认）
                embedder={
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-3-small"
                    }
                } if os.getenv("OPENAI_API_KEY") else None
            )
            
            # 执行多智能体协作
            app_logger.info(f"启动{len(agents)}个CrewAI原生智能体协作执行任务")
            
            # 异步执行crew
            result = await asyncio.get_event_loop().run_in_executor(
                None, crew.kickoff
            )
            
            execution_time = time.time() - start_time
            
            # 保存到现有的记忆服务（兼容现有系统）
            await self._save_to_existing_memory(
                task, str(result), user_id, agent_id, session_id, 
                use_memory, model
            )
            
            # 统计使用的工具
            used_tools = []
            for agent in agents:
                if hasattr(agent, 'tools') and agent.tools:
                    used_tools.extend([tool.name if hasattr(tool, 'name') else str(type(tool).__name__) for tool in agent.tools])
            
            app_logger.info(f"CrewAI原生多智能体任务执行完成，耗时: {execution_time:.2f}秒")
            
            return {
                "success": True,
                "result": str(result),
                "agent_type": self.agent_type,
                "framework": "CrewAI Native",
                "tools_used": list(set(used_tools)),  # 去重
                "execution_time": execution_time,
                "agents_used": agents_config,
                "process_type": process_type,
                "agents_count": len(agents),
                "memory_enabled": use_memory,
                "features_used": {
                    "rag": any("rag" in tool_type for tool_type in self._get_used_tool_types(agents_config)),
                    "memory": use_memory,
                    "tools": len(used_tools) > 0,
                    "multi_agent": len(agents) > 1
                }
            }
            
        except Exception as e:
            app_logger.error(f"CrewAI原生多智能体执行失败: {e}")
            return {
                "success": False,
                "result": f"CrewAI原生多智能体执行失败: {str(e)}",
                "agent_type": self.agent_type,
                "framework": "CrewAI Native",
                "tools_used": [],
                "execution_time": time.time() - start_time,
                "error": str(e)
            }
        finally:
            # 刷新监控数据
            self.monitoring_service.flush()
    
    def _auto_select_crew_config(self, task: str) -> tuple:
        """根据任务自动选择crew配置"""
        task_lower = task.lower()
        
        # 研究分析类任务
        if any(keyword in task_lower for keyword in ["研究", "分析", "调研", "分析报告", "调查"]):
            return ["researcher", "analyst", "writer"], "sequential"
        
        # 文档处理类任务
        elif any(keyword in task_lower for keyword in ["文档", "pdf", "doc", "文件", "总结"]):
            return ["researcher", "analyst", "writer", "reviewer"], "sequential"
        
        # 创作类任务
        elif any(keyword in task_lower for keyword in ["写", "创作", "文章", "内容", "博客"]):
            return ["researcher", "writer", "reviewer"], "sequential"
        
        # 数据分析类任务
        elif any(keyword in task_lower for keyword in ["数据", "统计", "图表", "csv", "excel"]):
            return ["researcher", "analyst", "writer"], "sequential"
        
        # 复杂决策类任务
        elif any(keyword in task_lower for keyword in ["决策", "方案", "计划", "策略", "建议"]):
            return ["researcher", "analyst", "coordinator", "reviewer"], "sequential"
        
        # Web相关任务
        elif any(keyword in task_lower for keyword in ["网站", "网页", "爬取", "搜索", "在线"]):
            return ["researcher", "analyst", "writer"], "sequential"
        
        # 默认配置
        else:
            return ["researcher", "writer"], "sequential"
    
    def _create_crewai_agents(self, agents_config: List[str], llm) -> List[Agent]:
        """创建CrewAI原生智能体团队"""
        agents = []
        
        for agent_key in agents_config:
            if agent_key not in self.agent_configs:
                app_logger.warning(f"未知的智能体角色: {agent_key}")
                continue
            
            config = self.agent_configs[agent_key]
            
            # 获取该智能体的工具
            agent_tools = []
            for tool_category in config["tools"]:
                if tool_category in self.all_tools:
                    agent_tools.extend(self.all_tools[tool_category])
            
            # 创建CrewAI原生Agent
            agent = Agent(
                role=config["role"],
                goal=config["goal"],
                backstory=config["backstory"],
                tools=agent_tools,
                llm=llm,
                verbose=True,
                memory=True,  # 启用智能体级别的记忆
                allow_delegation=False,  # 避免任务委派的复杂性
                max_iter=5,  # 限制迭代次数
                max_execution_time=300,  # 限制执行时间（秒）
            )
            
            agents.append(agent)
            app_logger.info(f"创建CrewAI原生智能体: {config['role']} (工具数量: {len(agent_tools)})")
        
        return agents
    
    def _create_crewai_tasks(
        self, 
        main_task: str, 
        agents: List[Agent], 
        agents_config: List[str]
    ) -> List[Task]:
        """创建CrewAI原生任务分解"""
        tasks = []
        
        for i, (agent, agent_key) in enumerate(zip(agents, agents_config)):
            # 根据智能体类型创建专门的任务描述
            task_descriptions = {
                "researcher": {
                    "description": f"""
作为高级研究分析师，请针对以下任务进行深入研究：

任务：{main_task}

请执行以下步骤：
1. 使用搜索工具收集相关信息和最新数据
2. 使用RAG工具查找本地知识库中的相关资料
3. 分析收集到的信息，识别关键要点和趋势
4. 整理研究发现，为后续分析提供基础

请充分利用你的搜索工具、RAG工具和网络工具来获取全面的信息。
                    """.strip(),
                    "expected_output": "详细的研究报告，包含收集到的关键信息、数据来源和初步分析结论"
                },
                
                "analyst": {
                    "description": f"""
作为数据分析专家，请基于前面的研究结果，对以下任务进行深入分析：

任务：{main_task}

请执行以下步骤：
1. 使用分析工具处理研究数据
2. 使用RAG工具查找相关的分析模式和方法
3. 识别数据中的关键模式、趋势和见解
4. 生成数据驱动的结论和建议

请充分利用你的分析工具和RAG工具来提供深度见解。
                    """.strip(),
                    "expected_output": "综合分析报告，包含数据见解、趋势分析、关键发现和具体建议"
                },
                
                "writer": {
                    "description": f"""
作为内容创作专家，请将研究和分析结果整理成高质量的内容：

任务：{main_task}

请执行以下步骤：
1. 使用RAG工具查找优秀的写作模板和案例
2. 使用文档工具处理和整理内容结构
3. 将复杂的分析结果转化为清晰易懂的内容
4. 确保内容逻辑清晰、结构完整、表达准确

请充分利用你的文档工具和RAG工具来创作高质量的内容。
                    """.strip(),
                    "expected_output": "结构化的高质量内容，包含清晰的结论、建议和易于理解的表述"
                },
                
                "coordinator": {
                    "description": f"""
作为项目协调员，请协调和整合团队工作成果：

任务：{main_task}

请执行以下步骤：
1. 使用RAG工具查找项目管理最佳实践
2. 使用分析工具评估工作成果的一致性
3. 整合不同团队成员的贡献
4. 确保最终输出的协调性和完整性

请充分利用你的协调工具和分析工具来确保项目成功。
                    """.strip(),
                    "expected_output": "协调后的整合方案，包含统一的执行建议和行动计划"
                },
                
                "reviewer": {
                    "description": f"""
作为质量审查员，请审查和优化前面生成的所有内容：

任务：{main_task}

请执行以下步骤：
1. 使用RAG工具查找质量标准和最佳实践
2. 使用文档工具检查内容的完整性和准确性
3. 识别需要改进的地方并提出具体建议
4. 确保最终输出满足高质量标准

请充分利用你的审查工具和文档工具来确保质量。
                    """.strip(),
                    "expected_output": "经过优化的最终内容，确保准确性、完整性和高质量标准"
                }
            }
            
            # 获取任务配置，如果没有则使用通用配置
            task_config = task_descriptions.get(agent_key, {
                "description": f"处理以下任务：{main_task}。请使用你的专业技能和可用工具来完成任务。",
                "expected_output": "任务处理结果和相关建议"
            })
            
            # 创建CrewAI原生Task
            task = Task(
                description=task_config["description"],
                expected_output=task_config["expected_output"],
                agent=agent,
                tools=agent.tools,  # 明确指定工具
            )
            
            tasks.append(task)
            app_logger.info(f"创建CrewAI原生任务 {i+1}: {agent.role}")
        
        return tasks
    
    def _configure_memory(self, use_memory: bool, user_id: Optional[str], session_id: Optional[str]) -> Dict[str, Any]:
        """配置CrewAI记忆系统"""
        if not use_memory:
            return {}
        
        memory_config = {
            "memory": True,
            # 可以配置自定义存储路径
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small"
                }
            } if os.getenv("OPENAI_API_KEY") else None
        }
        
        return memory_config
    
    async def _save_to_existing_memory(
        self, 
        task: str, 
        result: str, 
        user_id: Optional[str], 
        agent_id: Optional[str], 
        session_id: Optional[str],
        use_memory: bool,
        model: str
    ):
        """保存到现有的记忆服务（兼容现有系统）"""
        if not use_memory or not self.memory_service.is_enabled():
            return
        
        try:
            messages = [
                {"role": "user", "content": task},
                {"role": "assistant", "content": result}
            ]
            
            metadata = {
                "task_type": "crewai_native_multi_agent", 
                "model": model,
                "framework": "CrewAI Native"
            }
            
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
    
    def _get_used_tool_types(self, agents_config: List[str]) -> List[str]:
        """获取使用的工具类型"""
        used_types = []
        for agent_key in agents_config:
            if agent_key in self.agent_configs:
                used_types.extend(self.agent_configs[agent_key]["tools"])
        return list(set(used_types))
    
    def get_agent_info(self) -> Dict[str, Any]:
        """获取Agent信息"""
        return {
            "agent_type": self.agent_type,
            "name": "CrewAI Native Multi-Agent System",
            "description": "基于CrewAI原生技术栈的多智能体协作系统，集成RAG、Memory、Tools等功能",
            "framework": "CrewAI Native",
            "supports_tools": True,
            "supports_memory": True,
            "supports_rag": True,
            "available_models": self.llm_service.get_available_models(),
            "available_agent_roles": list(self.agent_configs.keys()),
            "available_tool_categories": list(self.all_tools.keys()),
            "supported_processes": ["sequential", "hierarchical"],
            "features": {
                "native_crewai": True,
                "rag_integration": True,
                "memory_system": True,
                "custom_tools": True,
                "multi_agent_collaboration": True,
                "caching": True,
                "embeddings": True
            },
            "tool_statistics": {
                category: len(tools) for category, tools in self.all_tools.items()
            }
        }
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """获取可用工具信息"""
        tool_info = {}
        for category, tools in self.all_tools.items():
            tool_names = []
            for tool in tools:
                if hasattr(tool, 'name'):
                    tool_names.append(tool.name)
                else:
                    tool_names.append(type(tool).__name__)
            tool_info[category] = tool_names
        return tool_info
    
    def get_predefined_roles(self) -> Dict[str, Dict[str, Any]]:
        """获取预定义角色信息"""
        roles_info = {}
        for role_key, config in self.agent_configs.items():
            roles_info[role_key] = {
                "role": config["role"],
                "goal": config["goal"],
                "backstory": config["backstory"],
                "tool_categories": config["tools"],
                "available_tools": [
                    tool_name for category in config["tools"] 
                    for tool_name in self.get_available_tools().get(category, [])
                ]
            }
        return roles_info


# 创建全局CrewAI原生多智能体实例
crew_multi_agent = CrewMultiAgent() 