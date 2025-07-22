#!/usr/bin/env python3
"""
AI Agent Demo API 测试脚本
测试添加知识库和三个agent的接口功能
"""

import requests
import json
import time
from datetime import datetime


class APITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def print_separator(self, title):
        """打印分隔符"""
        print("\n" + "="*60)
        print(f"🧪 {title}")
        print("="*60)
        
    def print_result(self, test_name, success, response_data=None, error=None):
        """打印测试结果"""
        status = "✅ 成功" if success else "❌ 失败"
        print(f"\n{status} {test_name}")
        
        if success and response_data:
            print("📋 响应数据:")
            print(json.dumps(response_data, indent=2, ensure_ascii=False))
        elif error:
            print(f"❗ 错误信息: {error}")
            
    def test_health_check(self):
        """测试健康检查"""
        self.print_separator("健康检查")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                self.print_result("健康检查", True, response.json())
                return True
            else:
                self.print_result("健康检查", False, error=f"状态码: {response.status_code}")
                return False
        except Exception as e:
            self.print_result("健康检查", False, error=str(e))
            return False
    
    def test_add_knowledge_base(self):
        """测试添加知识库"""
        self.print_separator("测试添加知识库")
        
        # 测试单个文档添加
        print("\n📚 测试1: 添加单个文档")
        try:
            data = {
                "collection_name": "test_knowledge",
                "texts": [
                    "人工智能(AI)是计算机科学的一个分支，旨在创建能够模拟人类智能的机器。",
                    "机器学习是人工智能的一个子领域，通过算法让计算机从数据中学习模式。",
                    "深度学习是机器学习的一种方法，使用多层神经网络来处理复杂的数据。"
                ],
                "metadatas": [
                    {"source": "AI基础", "category": "定义"},
                    {"source": "ML基础", "category": "概念"},
                    {"source": "DL基础", "category": "技术"}
                ]
            }
            
            response = self.session.post(f"{self.base_url}/api/v1/rag/documents", json=data)
            if response.status_code == 200:
                self.print_result("添加单个文档", True, response.json())
            else:
                self.print_result("添加单个文档", False, error=f"状态码: {response.status_code}, 响应: {response.text}")
                
        except Exception as e:
            self.print_result("添加单个文档", False, error=str(e))
        
        # 测试批量文档添加
        print("\n📚 测试2: 批量添加文档")
        try:
            data = {
                "collection_name": "test_knowledge",
                "documents": [
                    {
                        "text": "Python是一种高级编程语言，广泛用于数据科学和AI开发。",
                        "metadata": {"source": "Python指南", "category": "编程语言"}
                    },
                    {
                        "text": "FastAPI是一个现代的Python Web框架，用于构建高性能的API。",
                        "metadata": {"source": "FastAPI文档", "category": "Web框架"}
                    }
                ]
            }
            
            response = self.session.post(f"{self.base_url}/api/v1/rag/batch_documents", json=data)
            if response.status_code == 200:
                self.print_result("批量添加文档", True, response.json())
            else:
                self.print_result("批量添加文档", False, error=f"状态码: {response.status_code}, 响应: {response.text}")
                
        except Exception as e:
            self.print_result("批量添加文档", False, error=str(e))
    
    def test_langchain_react_agent(self):
        """测试LangChain ReAct Agent"""
        self.print_separator("测试LangChain ReAct Agent")
        
        try:
            data = {
                "agent_type": "react",
                "task": "请搜索并分析一下人工智能的最新发展趋势",
                "user_id": "test_user_001",
                "session_id": f"session_{int(time.time())}",
                "agent_id": "react_agent_001",
                "model": "gpt-4o-mini",
                "max_iterations": 3,
                "use_memory": True,
                "tools": ["search", "calculator"]
            }
            
            response = self.session.post(f"{self.base_url}/api/v1/agents/execute", json=data)
            if response.status_code == 200:
                self.print_result("LangChain ReAct Agent", True, response.json())
            else:
                self.print_result("LangChain ReAct Agent", False, error=f"状态码: {response.status_code}, 响应: {response.text}")
                
        except Exception as e:
            self.print_result("LangChain ReAct Agent", False, error=str(e))
    
    def test_crew_multi_agent(self):
        """测试Crew 多智能体"""
        self.print_separator("测试Crew 多智能体")
        
        try:
            data = {
                "agent_type": "crew",
                "task": "研究并撰写一份关于机器学习在医疗领域应用的报告",
                "user_id": "test_user_002", 
                "session_id": f"session_{int(time.time())}",
                "agent_id": "crew_agent_001",
                "model": "gpt-4o-mini",
                "use_memory": True,
                "tools": ["search"],
                "metadata": {
                    "agents": ["researcher", "analyst", "writer"],
                    "process": "sequential",
                    "max_iterations": 3
                }
            }
            
            response = self.session.post(f"{self.base_url}/api/v1/agents/execute", json=data)
            if response.status_code == 200:
                self.print_result("Crew 多智能体", True, response.json())
            else:
                self.print_result("Crew 多智能体", False, error=f"状态码: {response.status_code}, 响应: {response.text}")
                
        except Exception as e:
            self.print_result("Crew 多智能体", False, error=str(e))
    
    def test_dify_agent(self):
        """测试Dify Agent - 蓝领招聘"""
        self.print_separator("测试Dify Agent - 蓝领招聘")
        
        try:
            data = {
                "query": "你好，我在苏州火车站附近，想找工作",
                "user": "test_user_003",  # 修正字段名：user_id -> user
                "conversation_id": None,  # 创建新对话
                "inputs": {}  # 蓝领招聘的固定参数已在agent中设置
            }
            
            response = self.session.post(f"{self.base_url}/api/v1/agents/dify/chat", json=data)
            if response.status_code == 200:
                self.print_result("Dify Agent - 蓝领招聘", True, response.json())
            else:
                self.print_result("Dify Agent - 蓝领招聘", False, error=f"状态码: {response.status_code}, 响应: {response.text}")
                
        except Exception as e:
            self.print_result("Dify Agent - 蓝领招聘", False, error=str(e))
    
    def test_rag_query(self):
        """测试RAG查询"""
        self.print_separator("测试RAG查询")
        
        try:
            # RAG查询接口使用form data格式
            data = {
                "query": "什么是人工智能？",
                "collection_name": "test_knowledge",
                "top_k": 3
            }
            
            response = self.session.post(f"{self.base_url}/api/v1/rag/query", data=data)
            if response.status_code == 200:
                self.print_result("RAG查询", True, response.json())
            else:
                self.print_result("RAG查询", False, error=f"状态码: {response.status_code}, 响应: {response.text}")
                
        except Exception as e:
            self.print_result("RAG查询", False, error=str(e))
    
    def run_all_tests(self):
        """运行所有测试"""
        print(f"\n🚀 开始AI Agent Demo API测试")
        print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 服务地址: {self.base_url}")
        
        # 健康检查
        if not self.test_health_check():
            print("\n❌ 服务不可用，终止测试")
            return
        
        # 测试知识库功能
        self.test_add_knowledge_base()
        
        # 等待一下，让文档处理完成
        print("\n⏳ 等待2秒，让文档处理完成...")
        time.sleep(2)
        
        # 测试RAG查询
        self.test_rag_query()
        
        # 测试三个Agent
        self.test_langchain_react_agent()
        
        time.sleep(1)  # 避免请求太频繁
        self.test_crew_multi_agent()
        
        time.sleep(1)
        self.test_dify_agent()
        
        # 测试总结
        self.print_separator("测试完成")
        print("🎉 所有API测试已完成！")
        print("📝 请查看上面的测试结果。")
        print("💡 如果某些测试失败，可能是服务正在处理或配置问题。")


if __name__ == "__main__":
    # 创建测试器实例
    tester = APITester()
    
    # 运行所有测试
    tester.run_all_tests() 