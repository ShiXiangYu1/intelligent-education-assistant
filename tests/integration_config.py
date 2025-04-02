#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
集成测试配置模块 - 智能教育助手系统

此模块提供集成测试所需的配置数据，包括测试环境路径、连接参数和模拟数据。
通过集中管理配置，便于测试环境的统一设置和维护。

作者: AI助手
创建日期: 2023-04-02
"""

import os
import random
import string
from datetime import datetime

# 基础配置
BASE_TEST_DIR = "./test_data"  # 测试数据根目录

# 知识服务测试配置
KNOWLEDGE_SERVICE_CONFIG = {
    "vector_db_path": os.path.join(BASE_TEST_DIR, "vector_db"),
    "vector_db_dimension": 768,
    "es_hosts": ["http://localhost:9200"],
    "es_index_name": "test_knowledge_items",
    "quality_threshold": 0.7,
    "filter_strictness": 0.8,
    "enable_curriculum_filter": True
}

# 用户服务测试配置
USER_SERVICE_CONFIG = {
    "storage_path": os.path.join(BASE_TEST_DIR, "users"),
    "auth_secret_key": "test-secret-key-for-integration",
    "token_expire_minutes": 60,
    "refresh_token_expire_days": 7,
    "password_min_length": 8
}

# 推荐引擎测试配置
RECOMMENDATION_ENGINE_CONFIG = {
    "forgetting_base_retention": 0.9,
    "forgetting_rate": 0.1,
    "practice_boost": 0.1,
    "default_priority_weights": {
        "mastery": 0.5,
        "time_since_last_practice": 0.3,
        "difficulty": 0.2
    }
}

# 课标体系测试配置
CURRICULUM_CONFIG = {
    "storage_path": os.path.join(BASE_TEST_DIR, "curriculum"),
    "relations_file": "knowledge_relations.json",
    "knowledge_points_file": "knowledge_points.json"
}

# 学习记录整合测试配置
LEARNING_INTEGRATION_CONFIG = {
    "sync_interval": 3600,  # 同步间隔（秒）
    "trend_days": 30,       # 趋势统计天数
    "enable_auto_sync": True
}

# 模拟数据生成函数

def generate_test_user(user_id=None, role="student"):
    """
    生成测试用户数据
    
    参数:
        user_id: 用户ID，如果为None则自动生成
        role: 用户角色，默认为'student'
        
    返回:
        包含用户数据的字典
    """
    if user_id is None:
        user_id = f"test_user_{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}"
    
    return {
        "user_id": user_id,
        "username": f"testuser_{user_id[-5:]}",
        "email": f"{user_id}@example.com",
        "password": "test_password",
        "role": role,
        "grade": random.randint(1, 12) if role == "student" else None,
        "subject_preferences": random.sample(["数学", "语文", "英语", "物理", "化学", "生物", "历史", "地理"], 
                                         random.randint(1, 3)),
        "created_at": int(datetime.now().timestamp())
    }

def generate_knowledge_point(kp_id=None, subject="数学", grade=5):
    """
    生成测试知识点数据
    
    参数:
        kp_id: 知识点ID，如果为None则自动生成
        subject: 学科，默认为'数学'
        grade: 年级，默认为5
        
    返回:
        包含知识点数据的字典
    """
    if kp_id is None:
        kp_id = f"kp_{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}"
    
    # 数学知识点示例
    math_points = {
        "小学": ["数的认识", "加减法", "乘除法", "分数", "小数", "百分数", "图形与几何", "统计与概率"],
        "初中": ["整式", "方程", "不等式", "函数", "三角形", "四边形", "圆", "统计与概率"],
        "高中": ["集合", "函数", "三角函数", "平面向量", "数列", "概率统计", "立体几何", "解析几何"]
    }
    
    # 语文知识点示例
    chinese_points = {
        "小学": ["拼音", "汉字", "词语", "句子", "段落", "阅读理解", "写作"],
        "初中": ["文言文", "现代文", "诗词", "散文", "小说", "写作技巧"],
        "高中": ["文学常识", "语言表达", "文言文阅读", "古诗词鉴赏", "名篇名著", "议论文写作"]
    }
    
    # 英语知识点示例
    english_points = {
        "小学": ["字母", "单词", "句型", "时态", "阅读", "听力"],
        "初中": ["词汇", "语法", "阅读理解", "听力", "写作", "口语"],
        "高中": ["高级词汇", "复杂语法", "阅读技巧", "写作手法", "听说能力"]
    }
    
    # 根据年级确定学段
    if grade <= 6:
        level = "小学"
    elif grade <= 9:
        level = "初中"
    else:
        level = "高中"
    
    # 根据学科选择知识点集合
    if subject == "数学":
        point_set = math_points[level]
    elif subject == "语文":
        point_set = chinese_points[level]
    elif subject == "英语":
        point_set = english_points[level]
    else:
        point_set = ["基础知识", "核心概念", "重要原理", "应用技能"]
    
    # 随机选择知识点内容
    content = random.choice(point_set)
    
    return {
        "knowledge_point_id": kp_id,
        "title": content,
        "description": f"{subject}{grade}年级知识点：{content}",
        "subject": subject,
        "grade": grade,
        "difficulty": round(random.uniform(0.1, 1.0), 2),
        "importance": round(random.uniform(0.3, 1.0), 2),
        "keywords": [content] + random.sample(["基础", "重点", "难点", "考点"], random.randint(1, 2)),
        "created_at": int(datetime.now().timestamp())
    }

def generate_practice_record(user_id, knowledge_point_id, correct=None, timestamp=None):
    """
    生成测试练习记录
    
    参数:
        user_id: 用户ID
        knowledge_point_id: 知识点ID
        correct: 是否正确，如果为None则随机生成
        timestamp: 时间戳，如果为None则使用当前时间
        
    返回:
        包含练习记录的字典
    """
    if correct is None:
        correct = random.choice([True, False])
    
    if timestamp is None:
        timestamp = int(datetime.now().timestamp())
    
    return {
        "user_id": user_id,
        "knowledge_point_id": knowledge_point_id,
        "timestamp": timestamp,
        "correct": correct,
        "time_spent": random.randint(30, 300),  # 30秒到5分钟
        "score": random.randint(60, 100) if correct else random.randint(0, 59),
        "difficulty_feedback": random.choice(["简单", "适中", "困难"]),
        "attempt_count": random.randint(1, 3)
    }

def generate_knowledge_relations(knowledge_points):
    """
    为知识点生成关系数据
    
    参数:
        knowledge_points: 知识点列表
        
    返回:
        知识点关系词典，格式为{kp_id: [related_kp_ids]}
    """
    relations = {}
    kp_ids = [kp["knowledge_point_id"] for kp in knowledge_points]
    
    for kp in knowledge_points:
        kp_id = kp["knowledge_point_id"]
        
        # 筛选同学科、相近年级的知识点作为相关知识点
        related_candidates = [
            k["knowledge_point_id"] for k in knowledge_points
            if k["subject"] == kp["subject"] and 
               abs(k["grade"] - kp["grade"]) <= 1 and
               k["knowledge_point_id"] != kp_id
        ]
        
        # 随机选择1-3个相关知识点
        related_count = min(len(related_candidates), random.randint(1, 3))
        related_kps = random.sample(related_candidates, related_count) if related_count > 0 else []
        
        relations[kp_id] = related_kps
    
    return relations

# 测试数据集创建函数

def create_test_knowledge_dataset(count=10):
    """
    创建测试知识点数据集
    
    参数:
        count: 知识点数量
        
    返回:
        知识点列表和知识点关系词典
    """
    knowledge_points = []
    subjects = ["数学", "语文", "英语"]
    grades = list(range(1, 13))  # 1-12年级
    
    for i in range(count):
        kp_id = f"kp_{i:03d}"
        subject = random.choice(subjects)
        grade = random.choice(grades)
        kp = generate_knowledge_point(kp_id, subject, grade)
        knowledge_points.append(kp)
    
    relations = generate_knowledge_relations(knowledge_points)
    
    return knowledge_points, relations

def create_test_user_dataset(count=5):
    """
    创建测试用户数据集
    
    参数:
        count: 用户数量
        
    返回:
        用户列表
    """
    users = []
    
    for i in range(count):
        user_id = f"test_user_{i:03d}"
        role = "student" if i < count-1 else "teacher"  # 最后一个用户是教师
        user = generate_test_user(user_id, role)
        users.append(user)
    
    return users

def create_test_practice_dataset(users, knowledge_points, records_per_user=10):
    """
    创建测试练习记录数据集
    
    参数:
        users: 用户列表
        knowledge_points: 知识点列表
        records_per_user: 每个用户的记录数量
        
    返回:
        练习记录列表
    """
    practice_records = []
    kp_ids = [kp["knowledge_point_id"] for kp in knowledge_points]
    
    for user in users:
        if user["role"] != "student":
            continue
            
        # 为每个学生用户创建练习记录
        for _ in range(records_per_user):
            kp_id = random.choice(kp_ids)
            timestamp = int(datetime.now().timestamp()) - random.randint(0, 30*24*60*60)  # 最近30天内
            record = generate_practice_record(user["user_id"], kp_id, timestamp=timestamp)
            practice_records.append(record)
    
    # 按时间戳排序
    practice_records.sort(key=lambda x: x["timestamp"])
    
    return practice_records 