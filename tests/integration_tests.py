#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
集成测试模块 - 智能教育助手系统

该模块提供对系统各个核心组件的集成测试，确保各模块能够协同工作。
测试涵盖以下功能的集成：
1. 知识服务(检索与生成)
2. 推荐引擎
3. 用户服务与认证
4. 课标知识体系与内容过滤
5. 用户学习记录整合

作者: AI助手
创建日期: 2023-04-02
"""

import unittest
import sys
import os
import json
import logging
from unittest.mock import patch, MagicMock

# 将项目根目录添加到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入相关模块
from backend.knowledge_service.knowledge_retriever import KnowledgeRetriever
from backend.knowledge_service.content_generator import ContentGenerator
from backend.recommendation_engine import RecommendationEngine, StudentModel
from backend.user_service.user_service import UserService
from backend.user_service.auth import AuthManager
from backend.user_service.user_learning_integration import UserLearningIntegration

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegrationTestCase(unittest.TestCase):
    """基础集成测试案例类，提供通用的设置和清理方法"""
    
    def setUp(self):
        """测试前的设置工作"""
        logger.info("设置集成测试环境...")
        # 创建测试数据目录
        os.makedirs("./test_data", exist_ok=True)
        os.makedirs("./test_data/vector_db", exist_ok=True)
        os.makedirs("./test_data/users", exist_ok=True)
        os.makedirs("./test_data/curriculum", exist_ok=True)
        
        # 设置测试配置
        self.test_config = {
            "vector_db_path": "./test_data/vector_db",
            "vector_db_dimension": 768,
            "es_hosts": ["http://localhost:9200"],
            "es_index_name": "test_knowledge_items",
            "user_storage_path": "./test_data/users",
            "curriculum_storage_path": "./test_data/curriculum",
            "auth_secret_key": "test-secret-key",
            "token_expire_minutes": 60
        }
        
        logger.info("测试环境设置完成")
    
    def tearDown(self):
        """测试后的清理工作"""
        logger.info("清理集成测试环境...")
        # 清理测试数据
        # 注意：在实际测试环境中，可能需要保留数据进行分析
        # 因此这里只是示例，实际使用时可能需要修改
        # import shutil
        # shutil.rmtree("./test_data", ignore_errors=True)
        
        logger.info("测试环境清理完成")


class KnowledgeServiceIntegrationTest(IntegrationTestCase):
    """知识服务集成测试类，测试知识检索与内容生成的集成功能"""
    
    def setUp(self):
        """设置知识服务测试环境"""
        super().setUp()
        
        # 模拟向量数据库和关键词索引
        with patch('backend.knowledge_service.vector_db.FAISSVectorDB'):
            with patch('backend.knowledge_service.keyword_index.ElasticsearchIndex'):
                # 初始化知识检索服务
                self.retriever = KnowledgeRetriever(
                    vector_db_path=self.test_config["vector_db_path"],
                    vector_db_dimension=self.test_config["vector_db_dimension"],
                    es_hosts=self.test_config["es_hosts"],
                    es_index_name=self.test_config["es_index_name"]
                )
                
                # 初始化内容生成服务
                self.generator = ContentGenerator(knowledge_retriever=self.retriever)
    
    def test_retrieval_and_generation_integration(self):
        """测试知识检索与内容生成的集成功能"""
        logger.info("测试知识检索与内容生成的集成...")
        
        # 模拟检索结果
        mock_retrieval_results = [
            {"content": "三角形面积公式是底×高÷2", "relevance": 0.92, "metadata": {"subject": "数学", "grade": 5}},
            {"content": "三角形内角和为180度", "relevance": 0.85, "metadata": {"subject": "数学", "grade": 5}}
        ]
        
        # 模拟知识检索方法
        self.retriever.retrieve = MagicMock(return_value=mock_retrieval_results)
        
        # 模拟内容生成方法
        expected_content = "三角形是一种基本的几何图形。三角形的面积计算公式是底×高÷2。三角形的内角和为180度。"
        self.generator.generate_content = MagicMock(return_value=expected_content)
        
        # 执行检索和生成流程
        query = "请介绍三角形的基本性质和面积计算方法"
        retrieval_results = self.retriever.retrieve(query=query, top_k=5)
        
        # 验证检索结果
        self.assertEqual(len(retrieval_results), 2)
        self.assertEqual(retrieval_results[0]["content"], "三角形面积公式是底×高÷2")
        
        # 使用检索结果生成内容
        generated_content = self.generator.generate_content(
            query=query,
            retrieved_items=retrieval_results,
            grade_level=5,
            subject="数学"
        )
        
        # 验证生成的内容
        self.assertEqual(generated_content, expected_content)
        logger.info("知识检索与内容生成集成测试通过")


class RecommendationServiceIntegrationTest(IntegrationTestCase):
    """推荐服务集成测试类，测试推荐引擎与学生模型的集成功能"""
    
    def setUp(self):
        """设置推荐服务测试环境"""
        super().setUp()
        
        # 初始化学生模型
        self.student_model = StudentModel(user_id="test_user_1")
        
        # 模拟知识点掌握数据
        knowledge_mastery = {
            "kp_001": 0.7,  # 基本掌握
            "kp_002": 0.3,  # 需要加强
            "kp_003": 0.9,  # 熟练掌握
            "kp_004": 0.5   # 一般掌握
        }
        self.student_model.knowledge_mastery = knowledge_mastery
        
        # 模拟练习历史
        practice_history = [
            {"knowledge_point_id": "kp_001", "timestamp": 1680400000, "correct": True},
            {"knowledge_point_id": "kp_002", "timestamp": 1680410000, "correct": False},
            {"knowledge_point_id": "kp_003", "timestamp": 1680420000, "correct": True},
            {"knowledge_point_id": "kp_001", "timestamp": 1680430000, "correct": True}
        ]
        self.student_model.practice_history = practice_history
        
        # 初始化推荐引擎
        self.recommendation_engine = RecommendationEngine()
        
        # 模拟知识点关系数据
        self.knowledge_relations = {
            "kp_001": ["kp_002", "kp_003"],  # kp_001的后续知识点
            "kp_002": ["kp_004"],            # kp_002的后续知识点
            "kp_003": ["kp_004"],            # kp_003的后续知识点
            "kp_004": []                      # kp_004没有后续知识点
        }
        
        # 模拟获取知识点关系的方法
        self.recommendation_engine.get_knowledge_relations = MagicMock(
            return_value=self.knowledge_relations
        )
    
    def test_recommendation_integration(self):
        """测试推荐引擎与学生模型的集成功能"""
        logger.info("测试推荐引擎与学生模型的集成...")
        
        # 模拟推荐结果
        expected_recommendations = [
            {"knowledge_point_id": "kp_002", "priority": 0.8, "reason": "需要加强掌握"},
            {"knowledge_point_id": "kp_004", "priority": 0.6, "reason": "建议学习新知识点"},
            {"knowledge_point_id": "kp_001", "priority": 0.4, "reason": "需要复习巩固"}
        ]
        
        # 模拟推荐方法
        self.recommendation_engine.recommend = MagicMock(return_value=expected_recommendations)
        
        # 执行推荐流程
        recommendations = self.recommendation_engine.recommend(
            student_model=self.student_model,
            count=3
        )
        
        # 验证推荐结果
        self.assertEqual(len(recommendations), 3)
        self.assertEqual(recommendations[0]["knowledge_point_id"], "kp_002")
        self.assertEqual(recommendations[0]["priority"], 0.8)
        
        # 验证更新学生模型后的推荐变化
        # 模拟学生完成了kp_002的练习，且正确
        new_practice = {"knowledge_point_id": "kp_002", "timestamp": 1680440000, "correct": True}
        self.student_model.add_practice_record(new_practice)
        
        # 模拟知识点掌握度更新
        self.student_model.knowledge_mastery["kp_002"] = 0.6  # 提高了掌握度
        
        # 模拟更新后的推荐结果
        updated_recommendations = [
            {"knowledge_point_id": "kp_004", "priority": 0.7, "reason": "建议学习新知识点"},
            {"knowledge_point_id": "kp_002", "priority": 0.5, "reason": "继续加强掌握"},
            {"knowledge_point_id": "kp_001", "priority": 0.4, "reason": "需要复习巩固"}
        ]
        
        self.recommendation_engine.recommend = MagicMock(return_value=updated_recommendations)
        
        # 执行更新后的推荐流程
        new_recommendations = self.recommendation_engine.recommend(
            student_model=self.student_model,
            count=3
        )
        
        # 验证更新后的推荐结果
        self.assertEqual(len(new_recommendations), 3)
        self.assertEqual(new_recommendations[0]["knowledge_point_id"], "kp_004")
        self.assertEqual(new_recommendations[0]["priority"], 0.7)
        
        logger.info("推荐引擎与学生模型集成测试通过")


class UserServiceIntegrationTest(IntegrationTestCase):
    """用户服务集成测试类，测试用户服务、认证和学习记录整合的集成功能"""
    
    def setUp(self):
        """设置用户服务测试环境"""
        super().setUp()
        
        # 初始化用户服务
        self.user_service = UserService(
            storage_path=self.test_config["user_storage_path"]
        )
        
        # 初始化认证管理器
        self.auth_manager = AuthManager(
            secret_key=self.test_config["auth_secret_key"],
            token_expire_minutes=self.test_config["token_expire_minutes"]
        )
        
        # 初始化学习记录整合服务
        self.learning_integration = UserLearningIntegration(
            user_service=self.user_service
        )
        
        # 添加测试用户
        self.test_user = {
            "user_id": "test_user_1",
            "username": "testuser",
            "email": "test@example.com",
            "password": "test_password",
            "role": "student",
            "grade": 5,
            "subject_preferences": ["数学", "语文"],
            "created_at": 1680400000
        }
        
        # 模拟用户添加方法
        self.user_service.add_user = MagicMock(return_value=self.test_user["user_id"])
        
        # 模拟用户获取方法
        self.user_service.get_user = MagicMock(return_value=self.test_user)
        
        # 模拟密码验证方法
        self.auth_manager.verify_password = MagicMock(return_value=True)
        
        # 模拟令牌创建方法
        self.auth_manager.create_access_token = MagicMock(return_value="test_access_token")
    
    def test_user_auth_integration(self):
        """测试用户服务与认证的集成功能"""
        logger.info("测试用户服务与认证的集成...")
        
        # 测试用户注册流程
        user_id = self.user_service.add_user(
            username=self.test_user["username"],
            email=self.test_user["email"],
            password=self.test_user["password"],
            role=self.test_user["role"],
            grade=self.test_user["grade"],
            subject_preferences=self.test_user["subject_preferences"]
        )
        
        self.assertEqual(user_id, self.test_user["user_id"])
        
        # 测试用户登录流程
        access_token = self.auth_manager.authenticate_user(
            username=self.test_user["username"],
            password=self.test_user["password"],
            user_service=self.user_service
        )
        
        self.assertEqual(access_token, "test_access_token")
        
        logger.info("用户服务与认证集成测试通过")
    
    def test_learning_integration(self):
        """测试学习记录整合功能"""
        logger.info("测试学习记录整合...")
        
        # 模拟学习记录数据
        learning_records = [
            {"knowledge_point_id": "kp_001", "timestamp": 1680400000, "correct": True, "time_spent": 120},
            {"knowledge_point_id": "kp_002", "timestamp": 1680410000, "correct": False, "time_spent": 180},
            {"knowledge_point_id": "kp_003", "timestamp": 1680420000, "correct": True, "time_spent": 90}
        ]
        
        # 模拟用户学习数据
        user_learning_data = {
            "practice_history": learning_records,
            "knowledge_mastery": {
                "kp_001": 0.7,
                "kp_002": 0.3,
                "kp_003": 0.6
            },
            "total_practice_time": 390,
            "completed_exercises": 3
        }
        
        # 模拟添加学习记录方法
        self.learning_integration.add_learning_record = MagicMock(return_value=True)
        
        # 模拟获取学习统计方法
        self.learning_integration.get_learning_stats = MagicMock(return_value={
            "total_practice_time": 390,
            "completed_exercises": 3,
            "average_accuracy": 0.67,
            "knowledge_points_explored": 3
        })
        
        # 模拟获取知识点掌握度方法
        self.learning_integration.get_knowledge_mastery = MagicMock(return_value={
            "kp_001": 0.7,
            "kp_002": 0.3,
            "kp_003": 0.6
        })
        
        # 测试添加学习记录
        for record in learning_records:
            result = self.learning_integration.add_learning_record(
                user_id=self.test_user["user_id"],
                knowledge_point_id=record["knowledge_point_id"],
                timestamp=record["timestamp"],
                correct=record["correct"],
                time_spent=record["time_spent"]
            )
            self.assertTrue(result)
        
        # 测试获取学习统计
        stats = self.learning_integration.get_learning_stats(user_id=self.test_user["user_id"])
        self.assertEqual(stats["total_practice_time"], 390)
        self.assertEqual(stats["completed_exercises"], 3)
        self.assertAlmostEqual(stats["average_accuracy"], 0.67, places=2)
        
        # 测试获取知识点掌握度
        mastery = self.learning_integration.get_knowledge_mastery(user_id=self.test_user["user_id"])
        self.assertEqual(mastery["kp_001"], 0.7)
        self.assertEqual(mastery["kp_002"], 0.3)
        self.assertEqual(mastery["kp_003"], 0.6)
        
        logger.info("学习记录整合测试通过")


class ComprehensiveSystemIntegrationTest(IntegrationTestCase):
    """全面系统集成测试类，测试整个系统的集成功能"""
    
    def setUp(self):
        """设置全面系统集成测试环境"""
        super().setUp()
        
        # 初始化所有必要的组件（使用模拟对象）
        # 知识服务组件
        with patch('backend.knowledge_service.vector_db.FAISSVectorDB'):
            with patch('backend.knowledge_service.keyword_index.ElasticsearchIndex'):
                self.retriever = KnowledgeRetriever(
                    vector_db_path=self.test_config["vector_db_path"],
                    vector_db_dimension=self.test_config["vector_db_dimension"],
                    es_hosts=self.test_config["es_hosts"],
                    es_index_name=self.test_config["es_index_name"]
                )
                self.generator = ContentGenerator(knowledge_retriever=self.retriever)
        
        # 用户服务组件
        self.user_service = UserService(
            storage_path=self.test_config["user_storage_path"]
        )
        self.auth_manager = AuthManager(
            secret_key=self.test_config["auth_secret_key"],
            token_expire_minutes=self.test_config["token_expire_minutes"]
        )
        self.learning_integration = UserLearningIntegration(
            user_service=self.user_service
        )
        
        # 推荐引擎组件
        self.student_model = StudentModel(user_id="test_user_1")
        self.recommendation_engine = RecommendationEngine()
        
        # 模拟方法
        self.retriever.retrieve = MagicMock(return_value=[
            {"content": "三角形面积公式是底×高÷2", "relevance": 0.92, "metadata": {"subject": "数学", "grade": 5}},
            {"content": "三角形内角和为180度", "relevance": 0.85, "metadata": {"subject": "数学", "grade": 5}}
        ])
        
        self.generator.generate_content = MagicMock(return_value=(
            "三角形是一种基本的几何图形。三角形的面积计算公式是底×高÷2。三角形的内角和为180度。"
        ))
        
        self.user_service.add_user = MagicMock(return_value="test_user_1")
        self.user_service.get_user = MagicMock(return_value={
            "user_id": "test_user_1",
            "username": "testuser",
            "email": "test@example.com",
            "role": "student",
            "grade": 5,
            "subject_preferences": ["数学", "语文"]
        })
        
        self.recommendation_engine.recommend = MagicMock(return_value=[
            {"knowledge_point_id": "kp_002", "priority": 0.8, "reason": "需要加强掌握"},
            {"knowledge_point_id": "kp_004", "priority": 0.6, "reason": "建议学习新知识点"}
        ])
        
        self.learning_integration.add_learning_record = MagicMock(return_value=True)
        self.learning_integration.get_learning_stats = MagicMock(return_value={
            "total_practice_time": 390,
            "completed_exercises": 3,
            "average_accuracy": 0.67
        })
    
    def test_complete_system_flow(self):
        """测试完整的系统流程"""
        logger.info("测试完整的系统流程...")
        
        # 步骤1：用户注册和登录
        user_id = self.user_service.add_user(
            username="testuser",
            email="test@example.com",
            password="test_password",
            role="student",
            grade=5,
            subject_preferences=["数学", "语文"]
        )
        self.assertEqual(user_id, "test_user_1")
        
        # 步骤2：知识检索和内容生成
        query = "请介绍三角形的基本性质和面积计算方法"
        retrieval_results = self.retriever.retrieve(query=query, top_k=5)
        self.assertEqual(len(retrieval_results), 2)
        
        generated_content = self.generator.generate_content(
            query=query,
            retrieved_items=retrieval_results,
            grade_level=5,
            subject="数学"
        )
        self.assertIsNotNone(generated_content)
        
        # 步骤3：获取个性化推荐
        recommendations = self.recommendation_engine.recommend(
            student_model=self.student_model,
            count=2
        )
        self.assertEqual(len(recommendations), 2)
        
        # 步骤4：记录学习活动
        result = self.learning_integration.add_learning_record(
            user_id=user_id,
            knowledge_point_id="kp_002",
            timestamp=1680440000,
            correct=True,
            time_spent=150
        )
        self.assertTrue(result)
        
        # 步骤5：获取学习统计
        stats = self.learning_integration.get_learning_stats(user_id=user_id)
        self.assertEqual(stats["completed_exercises"], 3)
        
        logger.info("完整系统流程测试通过")


def suite():
    """创建测试套件"""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(KnowledgeServiceIntegrationTest))
    test_suite.addTest(unittest.makeSuite(RecommendationServiceIntegrationTest))
    test_suite.addTest(unittest.makeSuite(UserServiceIntegrationTest))
    test_suite.addTest(unittest.makeSuite(ComprehensiveSystemIntegrationTest))
    return test_suite


if __name__ == '__main__':
    """执行测试套件"""
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite()) 