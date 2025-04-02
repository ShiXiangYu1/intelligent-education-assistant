#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
知识服务API单元测试

该模块包含对知识服务API的全面测试，确保API接口正常工作。
测试覆盖以下方面:
1. 根路径和健康检查端点
2. 知识检索API功能
3. 内容生成API功能
4. 错误处理机制
"""

import unittest
import json
from unittest.mock import MagicMock, patch
import sys
import os
from typing import Dict, List, Optional, Any

# 添加项目根目录到Python路径，以便导入模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fastapi.testclient import TestClient

from backend.knowledge_service.api import app, get_retriever, get_generator
from backend.knowledge_service.knowledge_retriever import (
    KnowledgeRetriever, KnowledgeItem, SearchResult
)
from backend.knowledge_service.content_generator import (
    ContentGenerator, GeneratedContent, ContentSource
)


class TestKnowledgeServiceAPI(unittest.TestCase):
    """知识服务API测试类"""

    def setUp(self):
        """测试前的准备工作"""
        # 创建测试客户端
        self.client = TestClient(app)
        
        # 模拟知识检索器
        self.mock_retriever = MagicMock(spec=KnowledgeRetriever)
        
        # 模拟内容生成器
        self.mock_generator = MagicMock(spec=ContentGenerator)
        
        # 设置模拟的检索结果
        self.mock_search_results = [
            SearchResult(
                item=KnowledgeItem(
                    id="test1",
                    title="测试知识项1",
                    content="这是测试知识项1的内容",
                    grade_level=5,
                    subject="数学"
                ),
                score=0.9,
                keyword_score=0.8,
                vector_score=0.95
            )
        ]
        
        # 设置模拟的生成内容
        self.mock_generated_content = GeneratedContent(
            content="这是生成的内容",
            sources=[
                ContentSource(
                    id="source1",
                    title="来源1",
                    content_snippet="来源1的内容片段",
                    relevance_score=0.9
                )
            ],
            query="测试查询",
            grade_level=5,
            subject="数学",
            quality_score=0.85
        )
        
        # 配置模拟行为
        self.mock_retriever.retrieve.return_value = self.mock_search_results
        self.mock_generator.generate.return_value = self.mock_generated_content
        
        # 保存原始的依赖函数
        self.original_get_retriever = get_retriever
        self.original_get_generator = get_generator
        
        # 替换为模拟实例
        app.dependency_overrides[get_retriever] = lambda: self.mock_retriever
        app.dependency_overrides[get_generator] = lambda: self.mock_generator

    def tearDown(self):
        """测试后的清理工作"""
        # 恢复原始依赖
        app.dependency_overrides.clear()

    def test_root_endpoint(self):
        """测试根路径端点"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["service"], "智能教育助手 - 知识服务API")
        self.assertEqual(data["version"], "0.1.0")
        self.assertEqual(data["status"], "运行中")
        self.assertIn("/api/search - 知识检索", data["endpoints"])
        self.assertIn("/api/generate - 内容生成", data["endpoints"])

    def test_health_check(self):
        """测试健康检查端点"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)

    def test_search_api(self):
        """测试知识检索API"""
        # 构建请求
        request_data = {
            "query": "测试查询",
            "grade_level": 5,
            "subject": "数学",
            "top_k": 10,
            "min_score": 0.6
        }
        
        # 发送请求
        response = self.client.post("/api/search", json=request_data)
        self.assertEqual(response.status_code, 200)
        
        # 验证检索器被正确调用
        self.mock_retriever.retrieve.assert_called_once_with(
            query="测试查询",
            grade_level=5,
            subject="数学",
            top_k=10,
            min_score=0.6
        )
        
        # 验证响应内容
        data = response.json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["query"], "测试查询")
        self.assertIn("search_time", data)
        
        # 验证返回的知识项
        item = data["items"][0]
        self.assertEqual(item["id"], "test1")
        self.assertEqual(item["title"], "测试知识项1")
        self.assertEqual(item["grade_level"], 5)
        self.assertEqual(item["subject"], "数学")
        self.assertEqual(item["score"], 0.9)

    def test_search_api_empty_results(self):
        """测试检索结果为空的情况"""
        # 设置模拟检索结果为空
        self.mock_retriever.retrieve.return_value = []
        
        # 构建请求
        request_data = {
            "query": "不存在的查询",
            "grade_level": 5
        }
        
        # 发送请求
        response = self.client.post("/api/search", json=request_data)
        self.assertEqual(response.status_code, 200)
        
        # 验证响应内容
        data = response.json()
        self.assertEqual(data["total"], 0)
        self.assertEqual(data["query"], "不存在的查询")
        self.assertEqual(len(data["items"]), 0)

    def test_search_api_error(self):
        """测试检索过程发生错误时的处理"""
        # 设置模拟检索抛出异常
        self.mock_retriever.retrieve.side_effect = Exception("模拟检索错误")
        
        # 构建请求
        request_data = {
            "query": "测试查询",
            "grade_level": 5
        }
        
        # 发送请求
        response = self.client.post("/api/search", json=request_data)
        self.assertEqual(response.status_code, 500)
        
        # 验证错误响应
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("模拟检索错误", data["detail"])

    def test_generate_api(self):
        """测试内容生成API"""
        # 构建请求
        request_data = {
            "query": "测试生成内容",
            "grade_level": 5,
            "subject": "数学",
            "max_length": 500,
            "temperature": 0.7,
            "format": "text"
        }
        
        # 发送请求
        response = self.client.post("/api/generate", json=request_data)
        self.assertEqual(response.status_code, 200)
        
        # 验证生成器被正确调用
        self.mock_generator.generate.assert_called_once()
        
        # 获取传递给生成器的请求参数
        call_args = self.mock_generator.generate.call_args[0][0]
        self.assertEqual(call_args.query, "测试生成内容")
        self.assertEqual(call_args.grade_level, 5)
        self.assertEqual(call_args.subject, "数学")
        self.assertEqual(call_args.max_length, 500)
        self.assertEqual(call_args.temperature, 0.7)
        self.assertEqual(call_args.format, "text")
        
        # 验证响应内容
        data = response.json()
        self.assertEqual(data["content"], "这是生成的内容")
        self.assertEqual(data["quality_score"], 0.85)
        self.assertIn("generation_time", data)
        
        # 验证返回的来源引用
        self.assertEqual(len(data["sources"]), 1)
        source = data["sources"][0]
        self.assertEqual(source["id"], "source1")
        self.assertEqual(source["title"], "来源1")
        self.assertEqual(source["relevance_score"], 0.9)

    def test_generate_api_with_style(self):
        """测试带有风格参数的内容生成"""
        # 构建请求
        request_data = {
            "query": "测试生成内容",
            "grade_level": 5,
            "subject": "数学",
            "style": "通俗易懂",
            "format": "markdown"
        }
        
        # 发送请求
        response = self.client.post("/api/generate", json=request_data)
        self.assertEqual(response.status_code, 200)
        
        # 验证生成器被正确调用，并传递了风格参数
        call_args = self.mock_generator.generate.call_args[0][0]
        self.assertEqual(call_args.style, "通俗易懂")
        self.assertEqual(call_args.format, "markdown")

    def test_generate_api_error(self):
        """测试生成过程发生错误时的处理"""
        # 设置模拟生成抛出异常
        self.mock_generator.generate.side_effect = Exception("模拟生成错误")
        
        # 构建请求
        request_data = {
            "query": "测试生成内容",
            "grade_level": 5
        }
        
        # 发送请求
        response = self.client.post("/api/generate", json=request_data)
        self.assertEqual(response.status_code, 500)
        
        # 验证错误响应
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("模拟生成错误", data["detail"])


if __name__ == '__main__':
    unittest.main() 