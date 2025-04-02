#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API层面集成测试模块 - 智能教育助手系统

该模块测试系统各个模块API的协同工作，确保API接口能够正确地交互和集成。
测试内容包括所有服务的API端点，以及多个服务间的数据流通。

作者: AI助手
创建日期: 2023-04-02
"""

import unittest
import json
import sys
import os
import logging
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# 将项目根目录添加到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入测试工具
from tests.integration_utils import IntegrationTestSuite, MockVectorDB, MockKeywordIndex

# 导入相关模块
from backend.knowledge_service.api import app as knowledge_app
from backend.recommendation_api import app as recommendation_app
from backend.user_service.api import app as user_app

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class APIIntegrationTest(IntegrationTestSuite):
    """API集成测试类，测试各个模块API的协同工作"""
    
    @classmethod
    def setUpClass(cls):
        """测试类开始前的设置"""
        super().setUpClass()
        
        # 创建测试客户端
        cls.knowledge_client = TestClient(knowledge_app)
        cls.recommendation_client = TestClient(recommendation_app)
        cls.user_client = TestClient(user_app)
        
        # 设置授权头
        cls.headers = {"Authorization": "Bearer test_token"}
        
        # 添加模拟用户数据
        cls.test_user = {
            "user_id": "test_user_001",
            "username": "testuser",
            "email": "test@example.com",
            "password": "test_password",
            "role": "student",
            "grade": 5,
            "subject_preferences": ["数学", "语文"],
            "created_at": 1680400000
        }
        
        # 模拟知识点数据
        cls.test_knowledge_point = {
            "knowledge_point_id": "kp_001",
            "title": "三角形的面积",
            "description": "数学5年级知识点：三角形的面积计算方法",
            "subject": "数学",
            "grade": 5,
            "difficulty": 0.6,
            "importance": 0.8,
            "keywords": ["三角形", "面积", "基础", "重点"],
            "created_at": 1680400000
        }
    
    def test_knowledge_service_api(self):
        """测试知识服务API"""
        logger.info("测试知识服务API...")
        
        # 测试健康检查
        response = self.knowledge_client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})
        
        # 测试知识检索API
        search_data = {
            "query": "三角形的面积怎么计算",
            "top_k": 5,
            "grade_level": 5,
            "subject": "数学"
        }
        
        # 模拟知识检索结果
        with patch('backend.knowledge_service.api.knowledge_retriever.retrieve') as mock_retrieve:
            mock_retrieve.return_value = [
                {"content": "三角形面积公式是底×高÷2", "relevance": 0.92, "metadata": {"subject": "数学", "grade": 5}},
                {"content": "三角形内角和为180度", "relevance": 0.85, "metadata": {"subject": "数学", "grade": 5}}
            ]
            
            response = self.knowledge_client.post("/api/search", json=search_data)
            self.assertEqual(response.status_code, 200)
            
            results = response.json()
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["content"], "三角形面积公式是底×高÷2")
        
        # 测试内容生成API
        generate_data = {
            "query": "请解释三角形的面积计算方法",
            "grade_level": 5,
            "subject": "数学",
            "include_retrieved_items": True
        }
        
        # 模拟内容生成结果
        with patch('backend.knowledge_service.api.content_generator.generate_content') as mock_generate:
            expected_content = "三角形的面积计算公式是底×高÷2。这是一个基本的几何知识。"
            mock_generate.return_value = expected_content
            
            response = self.knowledge_client.post("/api/generate", json=generate_data)
            self.assertEqual(response.status_code, 200)
            
            result = response.json()
            self.assertIn("content", result)
            self.assertEqual(result["content"], expected_content)
            
            # 如果包含检索结果，则应返回retrieved_items
            if generate_data["include_retrieved_items"]:
                self.assertIn("retrieved_items", result)
        
        logger.info("知识服务API测试通过")
    
    def test_user_service_api(self):
        """测试用户服务API"""
        logger.info("测试用户服务API...")
        
        # 测试健康检查
        response = self.user_client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})
        
        # 测试用户注册API
        register_data = {
            "username": "new_user",
            "email": "new_user@example.com",
            "password": "new_password",
            "role": "student",
            "grade": 6,
            "subject_preferences": ["英语", "数学"]
        }
        
        # 模拟用户注册
        with patch('backend.user_service.api.user_service.add_user') as mock_add_user:
            mock_add_user.return_value = "new_user_id_001"
            
            response = self.user_client.post("/api/auth/register", json=register_data)
            self.assertEqual(response.status_code, 201)
            
            result = response.json()
            self.assertIn("user_id", result)
            self.assertEqual(result["user_id"], "new_user_id_001")
        
        # 测试用户登录API
        login_data = {
            "username": "testuser",
            "password": "test_password"
        }
        
        # 模拟用户登录
        with patch('backend.user_service.api.auth_manager.authenticate_user') as mock_auth:
            mock_auth.return_value = "test_access_token"
            
            response = self.user_client.post("/api/auth/login", json=login_data)
            self.assertEqual(response.status_code, 200)
            
            result = response.json()
            self.assertIn("access_token", result)
            self.assertEqual(result["access_token"], "test_access_token")
            self.assertIn("token_type", result)
            self.assertEqual(result["token_type"], "bearer")
        
        # 模拟获取当前用户信息
        with patch('backend.user_service.api.get_current_user') as mock_current_user:
            mock_current_user.return_value = self.test_user
            
            response = self.user_client.get("/api/users/me", headers=self.headers)
            self.assertEqual(response.status_code, 200)
            
            result = response.json()
            self.assertEqual(result["username"], self.test_user["username"])
            self.assertEqual(result["email"], self.test_user["email"])
        
        logger.info("用户服务API测试通过")
    
    def test_recommendation_api(self):
        """测试推荐服务API"""
        logger.info("测试推荐服务API...")
        
        # 测试健康检查
        response = self.recommendation_client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})
        
        # 测试获取推荐API
        recommend_data = {
            "user_id": "test_user_001",
            "count": 3,
            "filters": {
                "subject": "数学",
                "grade": 5
            }
        }
        
        # 模拟推荐结果
        with patch('backend.recommendation_api.recommendation_engine.recommend_for_user') as mock_recommend:
            mock_recommend.return_value = [
                {"knowledge_point_id": "kp_001", "priority": 0.8, "reason": "需要加强掌握"},
                {"knowledge_point_id": "kp_002", "priority": 0.6, "reason": "建议学习新知识点"},
                {"knowledge_point_id": "kp_003", "priority": 0.4, "reason": "需要复习巩固"}
            ]
            
            response = self.recommendation_client.post("/api/recommend", json=recommend_data)
            self.assertEqual(response.status_code, 200)
            
            results = response.json()
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 3)
            self.assertEqual(results[0]["knowledge_point_id"], "kp_001")
            self.assertEqual(results[0]["priority"], 0.8)
        
        # 测试提交练习记录API
        practice_data = {
            "user_id": "test_user_001",
            "knowledge_point_id": "kp_001",
            "correct": True,
            "time_spent": 120,
            "difficulty_feedback": "适中"
        }
        
        # 模拟提交练习记录
        with patch('backend.recommendation_api.recommendation_engine.add_practice_record') as mock_practice:
            mock_practice.return_value = True
            
            response = self.recommendation_client.post("/api/practice", json=practice_data)
            self.assertEqual(response.status_code, 200)
            
            result = response.json()
            self.assertIn("success", result)
            self.assertTrue(result["success"])
        
        logger.info("推荐服务API测试通过")
    
    def test_cross_service_integration(self):
        """测试跨服务集成"""
        logger.info("测试跨服务集成...")
        
        # 1. 用户注册
        register_data = {
            "username": "cross_test_user",
            "email": "cross_test@example.com",
            "password": "test_password",
            "role": "student",
            "grade": 5,
            "subject_preferences": ["数学"]
        }
        
        new_user_id = "cross_user_001"
        
        # 模拟用户注册
        with patch('backend.user_service.api.user_service.add_user') as mock_add_user:
            mock_add_user.return_value = new_user_id
            
            response = self.user_client.post("/api/auth/register", json=register_data)
            self.assertEqual(response.status_code, 201)
            
            result = response.json()
            self.assertEqual(result["user_id"], new_user_id)
        
        # 2. 用户登录获取令牌
        login_data = {
            "username": "cross_test_user",
            "password": "test_password"
        }
        
        access_token = "cross_test_token"
        
        # 模拟用户登录
        with patch('backend.user_service.api.auth_manager.authenticate_user') as mock_auth:
            mock_auth.return_value = access_token
            
            response = self.user_client.post("/api/auth/login", json=login_data)
            self.assertEqual(response.status_code, 200)
            
            result = response.json()
            self.assertEqual(result["access_token"], access_token)
        
        # 3. 知识检索
        search_data = {
            "query": "三角形的面积怎么计算",
            "top_k": 3,
            "grade_level": 5,
            "subject": "数学"
        }
        
        search_results = [
            {"content": "三角形面积公式是底×高÷2", "relevance": 0.92, "metadata": {"subject": "数学", "grade": 5}},
            {"content": "三角形内角和为180度", "relevance": 0.85, "metadata": {"subject": "数学", "grade": 5}}
        ]
        
        # 模拟知识检索
        with patch('backend.knowledge_service.api.knowledge_retriever.retrieve') as mock_retrieve:
            mock_retrieve.return_value = search_results
            
            response = self.knowledge_client.post("/api/search", json=search_data, 
                                                 headers={"Authorization": f"Bearer {access_token}"})
            self.assertEqual(response.status_code, 200)
            
            results = response.json()
            self.assertEqual(len(results), 2)
        
        # 4. 获取推荐
        recommend_data = {
            "user_id": new_user_id,
            "count": 2,
            "filters": {
                "subject": "数学",
                "grade": 5
            }
        }
        
        recommend_results = [
            {"knowledge_point_id": "kp_001", "priority": 0.8, "reason": "需要加强掌握"},
            {"knowledge_point_id": "kp_002", "priority": 0.6, "reason": "建议学习新知识点"}
        ]
        
        # 模拟获取推荐
        with patch('backend.recommendation_api.recommendation_engine.recommend_for_user') as mock_recommend:
            mock_recommend.return_value = recommend_results
            
            response = self.recommendation_client.post("/api/recommend", json=recommend_data, 
                                                       headers={"Authorization": f"Bearer {access_token}"})
            self.assertEqual(response.status_code, 200)
            
            results = response.json()
            self.assertEqual(len(results), 2)
        
        # 5. 提交练习记录
        practice_data = {
            "user_id": new_user_id,
            "knowledge_point_id": "kp_001",
            "correct": True,
            "time_spent": 120,
            "difficulty_feedback": "适中"
        }
        
        # 模拟提交练习记录
        with patch('backend.recommendation_api.recommendation_engine.add_practice_record') as mock_practice:
            mock_practice.return_value = True
            
            response = self.recommendation_client.post("/api/practice", json=practice_data, 
                                                      headers={"Authorization": f"Bearer {access_token}"})
            self.assertEqual(response.status_code, 200)
            
            result = response.json()
            self.assertTrue(result["success"])
        
        # 6. 获取学习统计
        # 模拟获取学习统计
        with patch('backend.user_service.api.learning_integration.get_learning_stats') as mock_stats:
            mock_stats.return_value = {
                "total_practice_time": 120,
                "completed_exercises": 1,
                "average_accuracy": 1.0,
                "knowledge_points_explored": 1
            }
            
            response = self.user_client.get(f"/api/users/{new_user_id}/learning/stats", 
                                           headers={"Authorization": f"Bearer {access_token}"})
            self.assertEqual(response.status_code, 200)
            
            result = response.json()
            self.assertEqual(result["total_practice_time"], 120)
            self.assertEqual(result["completed_exercises"], 1)
            self.assertEqual(result["average_accuracy"], 1.0)
        
        # 7. 内容生成结合知识检索
        generate_data = {
            "query": "请解释三角形的面积计算方法",
            "grade_level": 5,
            "subject": "数学"
        }
        
        # 模拟知识检索和内容生成
        with patch('backend.knowledge_service.api.knowledge_retriever.retrieve') as mock_retrieve:
            mock_retrieve.return_value = search_results
            
            with patch('backend.knowledge_service.api.content_generator.generate_content') as mock_generate:
                expected_content = "三角形的面积计算公式是底×高÷2。这是一个基本的几何知识。"
                mock_generate.return_value = expected_content
                
                response = self.knowledge_client.post("/api/generate", json=generate_data, 
                                                     headers={"Authorization": f"Bearer {access_token}"})
                self.assertEqual(response.status_code, 200)
                
                result = response.json()
                self.assertEqual(result["content"], expected_content)
        
        logger.info("跨服务集成测试通过")


def suite():
    """创建测试套件"""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(APIIntegrationTest))
    return test_suite


if __name__ == '__main__':
    """执行测试套件"""
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite()) 