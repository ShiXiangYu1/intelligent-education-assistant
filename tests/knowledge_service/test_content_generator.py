#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
内容生成模块单元测试

该模块包含对ContentGenerator类的全面测试，确保内容生成功能正常工作。
测试覆盖以下方面:
1. 基本内容生成功能
2. 不同输入参数的处理
3. 检索结果为空的情况处理
4. 质量检查和重新生成
5. 上下文准备和提示词构建
"""

import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os
import time
import random
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到Python路径，以便导入模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.knowledge_service.content_generator import (
    ContentGenerator, GenerationRequest, GeneratedContent,
    ContentSource, create_content_generator
)
from backend.knowledge_service.knowledge_retriever import (
    KnowledgeRetriever, KnowledgeItem, SearchResult
)


class TestContentGenerator(unittest.TestCase):
    """内容生成器测试类"""

    def setUp(self):
        """测试前的准备工作"""
        # 创建知识检索器的Mock对象
        self.mock_retriever = MagicMock(spec=KnowledgeRetriever)
        
        # 创建内容生成器实例
        self.generator = ContentGenerator(
            retriever=self.mock_retriever,
            llm_model="test-model",
            max_context_length=2000,
            quality_threshold=0.75,
            enable_quality_check=True
        )
        
        # 创建一些测试用的检索结果
        self.test_search_results = [
            SearchResult(
                item=KnowledgeItem(
                    id="item1",
                    title="测试知识项1",
                    content="这是第一个测试知识项的内容",
                    grade_level=5,
                    subject="数学"
                ),
                score=0.9,
                keyword_score=0.8,
                vector_score=0.95
            ),
            SearchResult(
                item=KnowledgeItem(
                    id="item2",
                    title="测试知识项2",
                    content="这是第二个测试知识项的内容",
                    grade_level=5,
                    subject="数学"
                ),
                score=0.8,
                keyword_score=0.75,
                vector_score=0.85
            )
        ]
        
        # 创建测试用的生成请求
        self.test_request = GenerationRequest(
            query="什么是分数?",
            grade_level=5,
            subject="数学",
            max_length=500,
            temperature=0.7,
            format="text"
        )
        
        # 修复random.random方法，使其返回固定值，确保测试可重复
        self.original_random = random.random
        random.random = lambda: 0.5

    def tearDown(self):
        """测试后的清理工作"""
        # 恢复random.random方法
        random.random = self.original_random

    def test_generate_with_search_results(self):
        """测试有检索结果时的内容生成"""
        # 设置模拟检索结果
        self.mock_retriever.retrieve.return_value = self.test_search_results
        
        # 模拟_call_llm方法
        expected_content = "这是生成的内容"
        with patch.object(self.generator, '_call_llm', return_value=expected_content) as mock_call_llm, \
             patch.object(self.generator, '_check_quality', return_value=(0.8, None)) as mock_check_quality:
            
            # 执行生成
            result = self.generator.generate(self.test_request)
            
            # 验证方法调用
            self.mock_retriever.retrieve.assert_called_once_with(
                query="什么是分数?",
                grade_level=5,
                subject="数学",
                top_k=5,
                min_score=0.6
            )
            
            # 验证_call_llm调用
            self.assertEqual(mock_call_llm.call_count, 1)
            
            # 验证_check_quality调用
            mock_check_quality.assert_called_once_with(
                expected_content, self.test_request, self.test_search_results
            )
            
            # 验证结果
            self.assertIsInstance(result, GeneratedContent)
            self.assertEqual(result.content, expected_content)
            self.assertEqual(result.query, "什么是分数?")
            self.assertEqual(result.grade_level, 5)
            self.assertEqual(result.subject, "数学")
            self.assertEqual(result.quality_score, 0.8)
            self.assertIsNone(result.feedback)
            self.assertEqual(len(result.sources), 2)
            self.assertEqual(result.sources[0].id, "item1")
            self.assertEqual(result.sources[1].id, "item2")

    def test_generate_without_search_results(self):
        """测试无检索结果时的内容生成"""
        # 设置空的检索结果
        self.mock_retriever.retrieve.return_value = []
        
        # 模拟_call_llm方法
        expected_content = "这是基础回复内容"
        with patch.object(self.generator, '_call_llm', return_value=expected_content) as mock_call_llm:
            
            # 执行生成
            result = self.generator.generate(self.test_request)
            
            # 验证调用了_generate_basic_response
            mock_call_llm.assert_called_once()  # 应该只调用一次
            
            # 验证结果
            self.assertIsInstance(result, GeneratedContent)
            self.assertEqual(result.content, expected_content)
            self.assertEqual(len(result.sources), 0)  # 无来源引用

    def test_quality_check_and_regenerate(self):
        """测试质量检查和内容重新生成"""
        # 设置模拟检索结果
        self.mock_retriever.retrieve.return_value = self.test_search_results
        
        # 模拟第一次生成的内容质量不佳，第二次生成的质量良好
        with patch.object(self.generator, '_call_llm') as mock_call_llm, \
             patch.object(self.generator, '_check_quality') as mock_check_quality:
            
            # 设置模拟返回值
            mock_call_llm.side_effect = ["低质量内容", "高质量内容"]
            mock_check_quality.side_effect = [(0.6, "内容质量不足"), (0.85, None)]
            
            # 执行生成
            result = self.generator.generate(self.test_request)
            
            # 验证_call_llm被调用了两次
            self.assertEqual(mock_call_llm.call_count, 2)
            
            # 验证_check_quality被调用了两次
            self.assertEqual(mock_check_quality.call_count, 2)
            
            # 验证最终使用了高质量内容
            self.assertEqual(result.content, "高质量内容")
            self.assertEqual(result.quality_score, 0.85)
            self.assertIsNone(result.feedback)

    def test_disable_quality_check(self):
        """测试禁用质量检查的情况"""
        # 设置模拟检索结果
        self.mock_retriever.retrieve.return_value = self.test_search_results
        
        # 禁用质量检查
        self.generator.enable_quality_check = False
        
        # 模拟_call_llm方法
        expected_content = "这是生成的内容"
        with patch.object(self.generator, '_call_llm', return_value=expected_content) as mock_call_llm, \
             patch.object(self.generator, '_check_quality') as mock_check_quality:
            
            # 执行生成
            result = self.generator.generate(self.test_request)
            
            # 验证_check_quality没有被调用
            mock_check_quality.assert_not_called()
            
            # 验证结果
            self.assertEqual(result.content, expected_content)
            self.assertIsNone(result.quality_score)
            self.assertIsNone(result.feedback)

    def test_prepare_context(self):
        """测试上下文准备功能"""
        # 执行_prepare_context方法
        context, sources = self.generator._prepare_context(self.test_search_results)
        
        # 验证上下文内容
        expected_context = "[1] 测试知识项1\n这是第一个测试知识项的内容\n\n[2] 测试知识项2\n这是第二个测试知识项的内容\n"
        self.assertEqual(context, expected_context)
        
        # 验证来源列表
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0].id, "item1")
        self.assertEqual(sources[0].title, "测试知识项1")
        self.assertEqual(sources[0].content_snippet, "这是第一个测试知识项的内容")
        self.assertEqual(sources[0].relevance_score, 0.9)
        
        # 测试上下文长度限制
        # 创建一个超长内容的测试项目
        long_content = "长内容" * 1000  # 约6000字符
        long_result = SearchResult(
            item=KnowledgeItem(
                id="long",
                title="长内容项",
                content=long_content,
                grade_level=5,
                subject="数学"
            ),
            score=0.7
        )
        
        # 执行_prepare_context方法
        self.generator.max_context_length = 100
        context, _ = self.generator._prepare_context([long_result])
        
        # 验证上下文被截断
        self.assertLessEqual(len(context), 103)  # 100 + "..."
        self.assertTrue(context.endswith("..."))

    def test_build_system_prompt(self):
        """测试系统提示词构建"""
        # 测试完整参数
        system_prompt = self.generator._build_system_prompt(self.test_request)
        self.assertIn("5年级", system_prompt)
        self.assertIn("数学", system_prompt)
        self.assertIn("text", system_prompt)  # 格式
        
        # 测试缺少参数的情况
        request_without_grade = GenerationRequest(
            query="测试查询",
            subject="语文",
            style="简洁"
        )
        system_prompt = self.generator._build_system_prompt(request_without_grade)
        self.assertIn("适合的年级", system_prompt)
        self.assertIn("语文", system_prompt)
        self.assertIn("简洁", system_prompt)  # 风格

    def test_build_user_prompt(self):
        """测试用户提示词构建"""
        context = "这是测试上下文"
        user_prompt = self.generator._build_user_prompt(self.test_request, context)
        
        # 验证提示词包含关键元素
        self.assertIn("参考资料:", user_prompt)
        self.assertIn(context, user_prompt)
        self.assertIn("问题/要求:", user_prompt)
        self.assertIn(self.test_request.query, user_prompt)
        self.assertIn("5年级", user_prompt)
        self.assertIn("数学", user_prompt)

    def test_check_quality(self):
        """测试内容质量检查"""
        # 模拟内容和请求
        content = "测试内容"
        
        # 执行质量检查
        score, feedback = self.generator._check_quality(content, self.test_request, self.test_search_results)
        
        # 我们设置了random.random为0.5，所以score应该是0.7 + 0.5 * 0.2 = 0.8
        self.assertEqual(score, 0.8)
        self.assertIsNone(feedback)  # score >= 0.8 时没有反馈
        
        # 测试低质量情况
        # 修改模拟得分使其低于0.8
        with patch('random.random', return_value=0.3):  # 0.7 + 0.3 * 0.2 = 0.76
            score, feedback = self.generator._check_quality(content, self.test_request, self.test_search_results)
            self.assertEqual(score, 0.76)
            self.assertIsNotNone(feedback)  # 应该有反馈建议

    def test_create_content_generator_factory(self):
        """测试内容生成器工厂函数"""
        # 创建检索器的Mock对象
        mock_retriever = MagicMock(spec=KnowledgeRetriever)
        
        # 测试默认参数
        generator = create_content_generator(mock_retriever)
        self.assertEqual(generator.llm_model, "gpt-3.5-turbo")
        self.assertEqual(generator.max_context_length, 4000)
        self.assertEqual(generator.quality_threshold, 0.7)
        self.assertTrue(generator.enable_quality_check)
        
        # 测试自定义参数
        llm_config = {
            "model": "custom-model",
            "max_context_length": 2000
        }
        generator = create_content_generator(
            mock_retriever,
            llm_config=llm_config,
            quality_threshold=0.8,
            enable_quality_check=False
        )
        self.assertEqual(generator.llm_model, "custom-model")
        self.assertEqual(generator.max_context_length, 2000)
        self.assertEqual(generator.quality_threshold, 0.8)
        self.assertFalse(generator.enable_quality_check)


if __name__ == '__main__':
    unittest.main() 