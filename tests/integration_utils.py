#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
集成测试工具模块 - 智能教育助手系统

提供集成测试所需的辅助工具函数，包括测试数据创建、环境准备和清理、结果验证等功能。
这些工具函数使集成测试代码更简洁、更易维护。

作者: AI助手
创建日期: 2023-04-02
"""

import os
import json
import shutil
import logging
import unittest
from unittest.mock import patch, MagicMock

from tests.integration_config import (
    BASE_TEST_DIR,
    KNOWLEDGE_SERVICE_CONFIG,
    USER_SERVICE_CONFIG,
    RECOMMENDATION_ENGINE_CONFIG,
    CURRICULUM_CONFIG,
    create_test_knowledge_dataset,
    create_test_user_dataset,
    create_test_practice_dataset
)

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_test_environment():
    """
    设置测试环境，创建所需目录结构
    
    返回:
        测试数据目录路径
    """
    logger.info("设置测试环境...")
    
    # 创建测试数据目录
    os.makedirs(BASE_TEST_DIR, exist_ok=True)
    os.makedirs(KNOWLEDGE_SERVICE_CONFIG["vector_db_path"], exist_ok=True)
    os.makedirs(USER_SERVICE_CONFIG["storage_path"], exist_ok=True)
    os.makedirs(CURRICULUM_CONFIG["storage_path"], exist_ok=True)
    
    logger.info(f"创建测试数据目录: {BASE_TEST_DIR}")
    
    return BASE_TEST_DIR


def cleanup_test_environment(keep_data=False):
    """
    清理测试环境
    
    参数:
        keep_data: 是否保留测试数据（用于调试）
    """
    if not keep_data and os.path.exists(BASE_TEST_DIR):
        logger.info(f"清理测试数据目录: {BASE_TEST_DIR}")
        shutil.rmtree(BASE_TEST_DIR, ignore_errors=True)
    else:
        logger.info(f"保留测试数据目录: {BASE_TEST_DIR}")


def create_test_data():
    """
    创建完整的测试数据集
    
    返回:
        包含所有测试数据的字典
    """
    logger.info("生成测试数据...")
    
    # 创建知识点和关系数据
    knowledge_points, knowledge_relations = create_test_knowledge_dataset(count=20)
    
    # 创建用户数据
    users = create_test_user_dataset(count=5)
    
    # 创建练习记录数据
    practice_records = create_test_practice_dataset(users, knowledge_points, records_per_user=10)
    
    # 保存测试数据
    test_data = {
        "knowledge_points": knowledge_points,
        "knowledge_relations": knowledge_relations,
        "users": users,
        "practice_records": practice_records
    }
    
    # 保存到文件
    save_test_data(test_data)
    
    logger.info(f"创建了 {len(knowledge_points)} 个知识点, {len(users)} 个用户, {len(practice_records)} 条练习记录")
    
    return test_data


def save_test_data(test_data):
    """
    将测试数据保存到文件
    
    参数:
        test_data: 测试数据字典
    """
    # 保存知识点数据
    knowledge_points_file = os.path.join(CURRICULUM_CONFIG["storage_path"], 
                                         CURRICULUM_CONFIG["knowledge_points_file"])
    with open(knowledge_points_file, 'w', encoding='utf-8') as f:
        json.dump(test_data["knowledge_points"], f, ensure_ascii=False, indent=2)
    
    # 保存知识点关系数据
    relations_file = os.path.join(CURRICULUM_CONFIG["storage_path"], 
                                 CURRICULUM_CONFIG["relations_file"])
    with open(relations_file, 'w', encoding='utf-8') as f:
        json.dump(test_data["knowledge_relations"], f, ensure_ascii=False, indent=2)
    
    # 保存用户数据
    users_file = os.path.join(USER_SERVICE_CONFIG["storage_path"], "users.json")
    with open(users_file, 'w', encoding='utf-8') as f:
        json.dump(test_data["users"], f, ensure_ascii=False, indent=2)
    
    # 保存练习记录数据
    practice_file = os.path.join(USER_SERVICE_CONFIG["storage_path"], "practice_records.json")
    with open(practice_file, 'w', encoding='utf-8') as f:
        json.dump(test_data["practice_records"], f, ensure_ascii=False, indent=2)
    
    logger.info("测试数据已保存到文件")


def load_test_data():
    """
    从文件加载测试数据
    
    返回:
        包含所有测试数据的字典
    """
    test_data = {}
    
    # 加载知识点数据
    knowledge_points_file = os.path.join(CURRICULUM_CONFIG["storage_path"], 
                                         CURRICULUM_CONFIG["knowledge_points_file"])
    if os.path.exists(knowledge_points_file):
        with open(knowledge_points_file, 'r', encoding='utf-8') as f:
            test_data["knowledge_points"] = json.load(f)
    
    # 加载知识点关系数据
    relations_file = os.path.join(CURRICULUM_CONFIG["storage_path"], 
                                 CURRICULUM_CONFIG["relations_file"])
    if os.path.exists(relations_file):
        with open(relations_file, 'r', encoding='utf-8') as f:
            test_data["knowledge_relations"] = json.load(f)
    
    # 加载用户数据
    users_file = os.path.join(USER_SERVICE_CONFIG["storage_path"], "users.json")
    if os.path.exists(users_file):
        with open(users_file, 'r', encoding='utf-8') as f:
            test_data["users"] = json.load(f)
    
    # 加载练习记录数据
    practice_file = os.path.join(USER_SERVICE_CONFIG["storage_path"], "practice_records.json")
    if os.path.exists(practice_file):
        with open(practice_file, 'r', encoding='utf-8') as f:
            test_data["practice_records"] = json.load(f)
    
    return test_data


class MockVectorDB:
    """模拟向量数据库类"""
    
    def __init__(self, vector_db_path, dimension):
        """
        初始化模拟向量数据库
        
        参数:
            vector_db_path: 向量数据库路径
            dimension: 向量维度
        """
        self.vector_db_path = vector_db_path
        self.dimension = dimension
        self.vectors = {}  # {id: vector}
        self.metadata = {}  # {id: metadata}
        logger.info(f"初始化模拟向量数据库，维度: {dimension}")
    
    def add_item(self, item_id, vector, metadata=None):
        """
        添加向量项
        
        参数:
            item_id: 项目ID
            vector: 向量数据
            metadata: 元数据
        """
        self.vectors[item_id] = vector
        self.metadata[item_id] = metadata or {}
        return True
    
    def search(self, query_vector, top_k=5):
        """
        搜索最相似的向量
        
        参数:
            query_vector: 查询向量
            top_k: 返回结果数量
            
        返回:
            (ids, scores)元组
        """
        # 简单模拟，返回前top_k个项目
        ids = list(self.vectors.keys())[:min(top_k, len(self.vectors))]
        scores = [0.9 - 0.1 * i for i in range(len(ids))]  # 模拟相似度得分
        return ids, scores
    
    def get_metadata(self, item_id):
        """
        获取项目元数据
        
        参数:
            item_id: 项目ID
            
        返回:
            元数据字典
        """
        return self.metadata.get(item_id, {})
    
    def save_index(self):
        """保存索引"""
        logger.info(f"保存向量索引到: {self.vector_db_path}")
        return True
    
    def load_index(self):
        """加载索引"""
        logger.info(f"从 {self.vector_db_path} 加载向量索引")
        return True


class MockKeywordIndex:
    """模拟关键词索引类"""
    
    def __init__(self, hosts, index_name):
        """
        初始化模拟关键词索引
        
        参数:
            hosts: ES主机列表
            index_name: 索引名称
        """
        self.hosts = hosts
        self.index_name = index_name
        self.documents = {}  # {id: document}
        logger.info(f"初始化模拟关键词索引: {index_name}")
    
    def add_document(self, doc_id, document):
        """
        添加文档
        
        参数:
            doc_id: 文档ID
            document: 文档内容
        """
        self.documents[doc_id] = document
        return True
    
    def search(self, query, filters=None, top_k=5):
        """
        搜索文档
        
        参数:
            query: 查询字符串
            filters: 过滤条件
            top_k: 返回结果数量
            
        返回:
            搜索结果列表
        """
        # 简单模拟，返回前top_k个文档
        results = []
        for doc_id, doc in list(self.documents.items())[:min(top_k, len(self.documents))]:
            # 应用过滤器
            if filters:
                skip = False
                for field, value in filters.items():
                    if field in doc and doc[field] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # 添加到结果
            results.append({
                "id": doc_id,
                "score": 0.8,  # 模拟相关性得分
                "document": doc
            })
        
        return results
    
    def delete_document(self, doc_id):
        """
        删除文档
        
        参数:
            doc_id: 文档ID
        """
        if doc_id in self.documents:
            del self.documents[doc_id]
            return True
        return False
    
    def create_index(self):
        """创建索引"""
        logger.info(f"创建索引: {self.index_name}")
        return True
    
    def delete_index(self):
        """删除索引"""
        logger.info(f"删除索引: {self.index_name}")
        self.documents = {}
        return True


def create_mock_patches():
    """
    创建模拟补丁，用于替换外部依赖
    
    返回:
        补丁列表
    """
    patches = [
        # 模拟向量数据库
        patch('backend.knowledge_service.vector_db.FAISSVectorDB', MockVectorDB),
        
        # 模拟关键词索引
        patch('backend.knowledge_service.keyword_index.ElasticsearchIndex', MockKeywordIndex),
        
        # 模拟OpenAI接口调用
        patch('backend.knowledge_service.content_generator.openai.ChatCompletion.create',
              return_value={
                  "choices": [
                      {
                          "message": {
                              "content": "这是由大模型生成的模拟内容，用于测试。"
                          }
                      }
                  ]
              })
    ]
    
    return patches


def apply_mock_patches(patches):
    """
    应用模拟补丁
    
    参数:
        patches: 补丁列表
        
    返回:
        启动的补丁列表
    """
    started_patches = [p.start() for p in patches]
    return started_patches


def stop_mock_patches(started_patches):
    """
    停止模拟补丁
    
    参数:
        started_patches: 启动的补丁列表
    """
    for p in started_patches:
        p.stop()


class IntegrationTestSuite(unittest.TestCase):
    """集成测试套件基类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类开始前的设置"""
        cls.test_dir = setup_test_environment()
        cls.test_data = create_test_data()
        
        # 创建和应用模拟补丁
        cls.patches = create_mock_patches()
        cls.started_patches = apply_mock_patches(cls.patches)
    
    @classmethod
    def tearDownClass(cls):
        """测试类结束后的清理"""
        # 停止模拟补丁
        stop_mock_patches(cls.started_patches)
        
        # 清理测试环境
        cleanup_test_environment(keep_data=False)
    
    def verify_knowledge_retrieval(self, retriever, query, expected_count=None):
        """
        验证知识检索结果
        
        参数:
            retriever: 知识检索器实例
            query: 查询字符串
            expected_count: 期望的结果数量
            
        返回:
            检索结果列表
        """
        results = retriever.retrieve(query=query, top_k=5)
        
        # 验证结果
        self.assertIsNotNone(results)
        self.assertIsInstance(results, list)
        
        if expected_count is not None:
            self.assertEqual(len(results), expected_count)
        
        # 验证结果项
        if results:
            self.assertIn('content', results[0])
            self.assertIn('relevance', results[0])
            self.assertIn('metadata', results[0])
        
        return results
    
    def verify_content_generation(self, generator, query, retrieved_items, grade_level, subject):
        """
        验证内容生成结果
        
        参数:
            generator: 内容生成器实例
            query: 查询字符串
            retrieved_items: 检索结果
            grade_level: 年级
            subject: 学科
            
        返回:
            生成的内容
        """
        content = generator.generate_content(
            query=query,
            retrieved_items=retrieved_items,
            grade_level=grade_level,
            subject=subject
        )
        
        # 验证生成的内容
        self.assertIsNotNone(content)
        self.assertIsInstance(content, str)
        self.assertTrue(len(content) > 0)
        
        return content
    
    def verify_user_creation(self, user_service, username, email, password, role="student"):
        """
        验证用户创建
        
        参数:
            user_service: 用户服务实例
            username: 用户名
            email: 邮箱
            password: 密码
            role: 角色
            
        返回:
            用户ID
        """
        user_id = user_service.add_user(
            username=username,
            email=email,
            password=password,
            role=role
        )
        
        # 验证用户ID
        self.assertIsNotNone(user_id)
        self.assertTrue(isinstance(user_id, str))
        
        # 验证用户数据
        user = user_service.get_user(user_id=user_id)
        self.assertEqual(user["username"], username)
        self.assertEqual(user["email"], email)
        self.assertEqual(user["role"], role)
        
        return user_id
    
    def verify_user_authentication(self, auth_manager, user_service, username, password):
        """
        验证用户认证
        
        参数:
            auth_manager: 认证管理器实例
            user_service: 用户服务实例
            username: 用户名
            password: 密码
            
        返回:
            访问令牌
        """
        access_token = auth_manager.authenticate_user(
            username=username,
            password=password,
            user_service=user_service
        )
        
        # 验证访问令牌
        self.assertIsNotNone(access_token)
        self.assertTrue(isinstance(access_token, str))
        
        return access_token
    
    def verify_recommendation(self, recommendation_engine, student_model, expected_count=None):
        """
        验证推荐结果
        
        参数:
            recommendation_engine: 推荐引擎实例
            student_model: 学生模型
            expected_count: 期望的推荐数量
            
        返回:
            推荐结果列表
        """
        recommendations = recommendation_engine.recommend(
            student_model=student_model,
            count=5
        )
        
        # 验证推荐结果
        self.assertIsNotNone(recommendations)
        self.assertIsInstance(recommendations, list)
        
        if expected_count is not None:
            self.assertEqual(len(recommendations), expected_count)
        
        # 验证推荐项
        if recommendations:
            self.assertIn('knowledge_point_id', recommendations[0])
            self.assertIn('priority', recommendations[0])
            self.assertIn('reason', recommendations[0])
        
        return recommendations 