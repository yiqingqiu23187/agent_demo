#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CrewAI原生多智能体系统演示脚本
演示RAG、Memory、Tools等功能
"""

import asyncio
import os
from app.agents.crew_multi_agent import crew_multi_agent


async def demo_crewai_native():
    """演示CrewAI原生多智能体系统"""
    
    print("🚀 CrewAI原生多智能体系统演示")
    print("=" * 50)
    
    # 1. 显示系统信息
    agent_info = crew_multi_agent.get_agent_info()
    print(f"📋 系统信息:")
    print(f"   - 框架: {agent_info['framework']}")
    print(f"   - 支持RAG: {'✅' if agent_info['supports_rag'] else '❌'}")
    print(f"   - 支持Memory: {'✅' if agent_info['supports_memory'] else '❌'}")
    print(f"   - 支持Tools: {'✅' if agent_info['supports_tools'] else '❌'}")
    print()
    
    # 2. 显示可用角色
    roles = crew_multi_agent.get_predefined_roles()
    print("🎭 可用的智能体角色:")
    for role_key, role_info in roles.items():
        print(f"   - {role_key}: {role_info['role']}")
        print(f"     目标: {role_info['goal']}")
        print(f"     工具类别: {', '.join(role_info['tool_categories'])}")
        print()
    
    # 3. 演示简单任务执行
    print("🔄 执行演示任务...")
    print("-" * 30)
    
    # 使用最基础的配置进行演示
    demo_task = "分析人工智能在教育领域的应用前景和挑战"
    
    try:
        result = await crew_multi_agent.execute(
            task=demo_task,
            user_id="demo_user",
            session_id="demo_session", 
            model="gpt-4o-mini",  # 使用较便宜的模型
            use_memory=True,
            crew_config={
                "agents": ["researcher", "analyst"],  # 只使用两个基础角色
                "process": "sequential"
            }
        )
        
        print("✅ 任务执行完成!")
        print(f"📊 执行结果:")
        print(f"   - 成功: {result['success']}")
        print(f"   - 框架: {result.get('framework', 'N/A')}")
        print(f"   - 执行时间: {result.get('execution_time', 0):.2f}秒")
        print(f"   - 使用的智能体: {', '.join(result.get('agents_used', []))}")
        print(f"   - 智能体数量: {result.get('agents_count', 0)}")
        print(f"   - 启用的功能:")
        features = result.get('features_used', {})
        for feature, enabled in features.items():
            print(f"     * {feature}: {'✅' if enabled else '❌'}")
        print()
        
        # 显示结果的前200个字符
        if result['success'] and result['result']:
            print("📝 执行结果预览:")
            preview = result['result'][:300] + "..." if len(result['result']) > 300 else result['result']
            print(f"   {preview}")
        
    except Exception as e:
        print(f"❌ 任务执行失败: {e}")
        print("💡 这可能是由于缺少API密钥或网络问题")
    
    print("\n🎉 演示完成!")
    print("📚 CrewAI原生多智能体系统支持的主要功能:")
    print("   • RAG (检索增强生成): 集成多种文档和数据搜索工具")
    print("   • Memory (记忆系统): 短期、长期和实体记忆")
    print("   • Tools (工具系统): 丰富的预置和自定义工具")
    print("   • Multi-Agent (多智能体): 多个专业角色协作")
    print("   • Native CrewAI: 使用原生CrewAI技术栈")


if __name__ == "__main__":
    # 设置环境变量（如果需要）
    # os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    
    print("🌟 启动CrewAI原生多智能体系统演示...")
    asyncio.run(demo_crewai_native()) 