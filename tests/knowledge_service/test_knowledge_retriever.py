#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
知识检索模块单元测试

该模块包含对KnowledgeRetriever类的全面测试，确保检索功能正常工作。
测试覆盖以下方面:
1. 检索器初始化参数验证
2. 基本检索功能测试
3. 参数边界条件测试
4. 过滤功能测试
5. 结果合并逻辑测试
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from typing import Dict, List, Optional

# 添加项目根目录到Python路径，以便导入模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.knowledge_service.knowledge_retriever import (
    KnowledgeRetriever, KnowledgeItem, SearchResult, create_retriever
)


class TestKnowledgeRetriever(unittest.TestCase):
    """知识检索器测试类"""

    def setUp(self):
        """测试前的准备工作"""
        # 创建一个基本的检索器实例用于测试
        self.retriever = KnowledgeRetriever(
            vector_db_client=MagicMock(),
            keyword_index=MagicMock(),
            vector_weight=0.7,
            keyword_weight=0.3,
            enable_grade_filter=True
        )
        
        # 创建一些测试用的知识项
        self.test_items = [
            KnowledgeItem(
                id="test1",
                title="测试知识项1",
                content="这是一个测试知识项，用于单元测试",
                grade_level=3,
                subject="数学",
                keywords=["测试", "数学", "单元测试"]
            ),
            KnowledgeItem(
                id="test2",
                title="高年级知识项",
                content="这是一个高年级的知识项，用于测试年级过滤",
                grade_level=8,
                subject="数学",
                keywords=["高年级", "数学"]
            ),
            KnowledgeItem(
                id="test3",
                title="不同学科知识项",
                content="这是一个不同学科的知识项，用于测试学科过滤",
                grade_level=3,
                subject="语文",
                keywords=["语文", "测试"]
            )
        ]

    def test_retriever_initialization(self):
        """测试检索器初始化"""
        # 测试默认参数
        retriever = KnowledgeRetriever()
        self.assertIsNone(retriever.vector_db_client)
        self.assertIsNone(retriever.keyword_index)
        self.assertEqual(retriever.vector_weight, 0.6)
        self.assertEqual(retriever.keyword_weight, 0.4)
        self.assertTrue(retriever.enable_grade_filter)
        
        # 测试自定义参数
        retriever = KnowledgeRetriever(
            vector_weight=0.8,
            keyword_weight=0.2,
            enable_grade_filter=False
        )
        self.assertEqual(retriever.vector_weight, 0.8)
        self.assertEqual(retriever.keyword_weight, 0.2)
        self.assertFalse(retriever.enable_grade_filter)
        
        # 测试权重归一化
        retriever = KnowledgeRetriever(
            vector_weight=2,
            keyword_weight=3
        )
        self.assertAlmostEqual(retriever.vector_weight, 0.4)
        self.assertAlmostEqual(retriever.keyword_weight, 0.6)
        self.assertEqual(retriever.vector_weight + retriever.keyword_weight, 1.0)

    def test_basic_retrieve(self):
        """测试基本检索功能"""
        # 模拟_keyword_search和_vector_search方法
        with patch.object(self.retriever, '_keyword_search') as mock_keyword_search, \
             patch.object(self.retriever, '_vector_search') as mock_vector_search, \
             patch.object(self.retriever, '_merge_results') as mock_merge_results:
            
            # 设置模拟返回值
            mock_keyword_results = [
                SearchResult(
                    item=self.test_items[0],
                    score=0.8,
                    keyword_score=0.8,
                    vector_score=0.0
                )
            ]
            mock_vector_results = [
                SearchResult(
                    item=self.test_items[0],
                    score=0.9,
                    keyword_score=0.0,
                    vector_score=0.9
                )
            ]
            merged_results = [
                SearchResult(
                    item=self.test_items[0],
                    score=0.85,  # (0.7*0.9 + 0.3*0.8)
                    keyword_score=0.8,
                    vector_score=0.9
                )
            ]
            
            mock_keyword_search.return_value = mock_keyword_results
            mock_vector_search.return_value = mock_vector_results
            mock_merge_results.return_value = merged_results
            
            # 执行检索
            results = self.retriever.retrieve(
                query="测试查询",
                grade_level=5,
                subject="数学",
                top_k=10,
                min_score=0.6
            )
            
            # 验证方法调用
            mock_keyword_search.assert_called_once_with("测试查询", 5, "数学", 20)
            mock_vector_search.assert_called_once_with("测试查询", 5, "数学", 20)
            mock_merge_results.assert_called_once_with(mock_keyword_results, mock_vector_results)
            
            # 验证结果
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].item.id, "test1")
            self.assertEqual(results[0].score, 0.85)

    def test_filter_by_grade(self):
        """测试年级过滤功能"""
        # 准备测试数据
        item1 = self.test_items[0]  # 年级3
        item2 = self.test_items[1]  # 年级8
        
        # 测试年级过滤启用
        self.retriever.enable_grade_filter = True
        
        # 年级5应该过滤掉年级8的项目
        self.assertTrue(self.retriever._apply_filters(item1, 5, None))
        self.assertFalse(self.retriever._apply_filters(item2, 5, None))
        
        # 年级10应该允许两个项目
        self.assertTrue(self.retriever._apply_filters(item1, 10, None))
        self.assertTrue(self.retriever._apply_filters(item2, 10, None))
        
        # 测试年级过滤禁用
        self.retriever.enable_grade_filter = False
        self.assertTrue(self.retriever._apply_filters(item1, 5, None))
        self.assertTrue(self.retriever._apply_filters(item2, 5, None))

    def test_filter_by_subject(self):
        """测试学科过滤功能"""
        # 准备测试数据
        item1 = self.test_items[0]  # 数学
        item3 = self.test_items[2]  # 语文
        
        # 测试学科过滤
        self.assertTrue(self.retriever._apply_filters(item1, None, "数学"))
        self.assertFalse(self.retriever._apply_filters(item3, None, "数学"))
        self.assertTrue(self.retriever._apply_filters(item3, None, "语文"))
        
        # 不指定学科时不应过滤
        self.assertTrue(self.retriever._apply_filters(item1, None, None))
        self.assertTrue(self.retriever._apply_filters(item3, None, None))

    def test_merge_results(self):
        """测试结果合并逻辑"""
        # 准备测试数据
        keyword_results = [
            SearchResult(
                item=self.test_items[0],
                score=0.8,
                keyword_score=0.8,
                vector_score=0.0
            ),
            SearchResult(
                item=self.test_items[2],
                score=0.7,
                keyword_score=0.7,
                vector_score=0.0
            )
        ]
        
        vector_results = [
            SearchResult(
                item=self.test_items[0],
                score=0.9,
                keyword_score=0.0,
                vector_score=0.9
            ),
            SearchResult(
                item=self.test_items[1],
                score=0.75,
                keyword_score=0.0,
                vector_score=0.75
            )
        ]
        
        # 设置检索器权重
        self.retriever.vector_weight = 0.7
        self.retriever.keyword_weight = 0.3
        
        # 执行合并
        merged = self.retriever._merge_results(keyword_results, vector_results)
        
        # 验证结果
        self.assertEqual(len(merged), 3)  # 应该有3个唯一结果
        
        # 按ID排序以便比较
        merged_by_id = {result.item.id: result for result in merged}
        
        # 验证test1(同时在关键词和向量结果中)
        self.assertIn("test1", merged_by_id)
        result1 = merged_by_id["test1"]
        self.assertEqual(result1.keyword_score, 0.8)
        self.assertEqual(result1.vector_score, 0.9)
        expected_score1 = 0.3 * 0.8 + 0.7 * 0.9  # 0.87
        self.assertAlmostEqual(result1.score, expected_score1)
        
        # 验证test2(仅在向量结果中)
        self.assertIn("test2", merged_by_id)
        result2 = merged_by_id["test2"]
        self.assertEqual(result2.keyword_score, 0.0)
        self.assertEqual(result2.vector_score, 0.75)
        expected_score2 = 0.7 * 0.75  # 0.525
        self.assertAlmostEqual(result2.score, expected_score2)
        
        # 验证test3(仅在关键词结果中)
        self.assertIn("test3", merged_by_id)
        result3 = merged_by_id["test3"]
        self.assertEqual(result3.keyword_score, 0.7)
        self.assertEqual(result3.vector_score, 0.0)
        expected_score3 = 0.3 * 0.7  # 0.21
        self.assertAlmostEqual(result3.score, expected_score3)

    def test_create_retriever_factory(self):
        """测试检索器工厂函数"""
        # 测试默认参数
        retriever = create_retriever()
        self.assertIsNone(retriever.vector_db_client)
        self.assertIsNone(retriever.keyword_index)
        self.assertEqual(retriever.vector_weight, 0.6)
        self.assertEqual(retriever.keyword_weight, 0.4)
        self.assertTrue(retriever.enable_grade_filter)
        
        # 测试自定义参数
        retriever = create_retriever(
            vector_weight=0.8,
            keyword_weight=0.2,
            enable_grade_filter=False
        )
        self.assertEqual(retriever.vector_weight, 0.8)
        self.assertEqual(retriever.keyword_weight, 0.2)
        self.assertFalse(retriever.enable_grade_filter)

    def test_empty_results(self):
        """测试空结果处理"""
        # 模拟空的关键词和向量搜索结果
        with patch.object(self.retriever, '_keyword_search', return_value=[]), \
             patch.object(self.retriever, '_vector_search', return_value=[]):
            
            results = self.retriever.retrieve("空查询")
            
            # 验证结果为空
            self.assertEqual(len(results), 0)

    def test_min_score_filtering(self):
        """测试最小分数过滤"""
        # 创建模拟结果
        merged_results = [
            SearchResult(item=self.test_items[0], score=0.9),
            SearchResult(item=self.test_items[1], score=0.7),
            SearchResult(item=self.test_items[2], score=0.5)
        ]
        
        # 模拟方法调用
        with patch.object(self.retriever, '_keyword_search', return_value=[]), \
             patch.object(self.retriever, '_vector_search', return_value=[]), \
             patch.object(self.retriever, '_merge_results', return_value=merged_results), \
             patch.object(self.retriever, '_apply_filters', return_value=True):
            
            # 测试不同的最小分数
            results1 = self.retriever.retrieve("测试查询", min_score=0.8)
            self.assertEqual(len(results1), 1)  # 只有一个结果>=0.8
            
            results2 = self.retriever.retrieve("测试查询", min_score=0.6)
            self.assertEqual(len(results2), 2)  # 有两个结果>=0.6
            
            results3 = self.retriever.retrieve("测试查询", min_score=0.4)
            self.assertEqual(len(results3), 3)  # 所有结果>=0.4


if __name__ == '__main__':
    unittest.main() 