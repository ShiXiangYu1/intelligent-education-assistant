#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用户数据模型

定义了系统中用户相关的数据模型，包括用户基本信息、用户角色和用户画像。
"""

import time
import enum
from typing import Dict, List, Optional, Set, Union, Any
from pydantic import BaseModel, Field, EmailStr, validator


class UserRole(str, enum.Enum):
    """用户角色枚举"""
    STUDENT = "student"  # 学生
    TEACHER = "teacher"  # 教师
    PARENT = "parent"    # 家长
    ADMIN = "admin"      # 管理员


class LearningRecord(BaseModel):
    """学习记录"""
    exercise_count: int = Field(0, description="练习题数量")
    correct_count: int = Field(0, description="正确题目数量")
    total_time: float = Field(0.0, description="总学习时间(分钟)")
    last_active: float = Field(default_factory=time.time, description="最后活跃时间")
    knowledge_points: Dict[str, float] = Field(
        default={}, description="知识点掌握情况，键为知识点ID，值为掌握程度(0-1)"
    )
    
    @property
    def accuracy_rate(self) -> float:
        """准确率"""
        if self.exercise_count == 0:
            return 0.0
        return self.correct_count / self.exercise_count


class UserProfile(BaseModel):
    """用户画像"""
    grade_level: Optional[int] = Field(None, description="年级(1-12)")
    subjects: List[str] = Field(default=[], description="关注的学科")
    learning_records: Dict[str, LearningRecord] = Field(
        default={}, description="学习记录，键为学科ID"
    )
    preferences: Dict[str, Any] = Field(default={}, description="学习偏好设置")
    learning_style: Optional[str] = Field(None, description="学习风格")
    learning_rate: float = Field(0.1, description="学习速率(0-1)")
    forgetting_rate: float = Field(0.05, description="遗忘速率(0-1)")
    
    # 可选的个性化属性
    favorite_topics: List[str] = Field(default=[], description="感兴趣的主题")
    weak_points: List[str] = Field(default=[], description="薄弱知识点")
    strong_points: List[str] = Field(default=[], description="擅长知识点")
    
    def update_learning_record(
        self, 
        subject: str, 
        is_correct: bool, 
        time_spent: float,
        knowledge_points: Dict[str, float] = None
    ) -> None:
        """
        更新学习记录
        
        Args:
            subject: 学科ID
            is_correct: 是否正确
            time_spent: 花费时间(分钟)
            knowledge_points: 知识点掌握更新，键为知识点ID，值为掌握程度
        """
        if subject not in self.learning_records:
            self.learning_records[subject] = LearningRecord()
        
        record = self.learning_records[subject]
        record.exercise_count += 1
        if is_correct:
            record.correct_count += 1
        record.total_time += time_spent
        record.last_active = time.time()
        
        # 更新知识点掌握情况
        if knowledge_points:
            for kp_id, mastery in knowledge_points.items():
                record.knowledge_points[kp_id] = mastery


class User(BaseModel):
    """用户模型"""
    id: str = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    email: EmailStr = Field(..., description="电子邮箱")
    hashed_password: str = Field(..., description="哈希密码")
    salt: str = Field(..., description="密码盐")
    role: UserRole = Field(UserRole.STUDENT, description="用户角色")
    profile: UserProfile = Field(default_factory=UserProfile, description="用户画像")
    real_name: Optional[str] = Field(None, description="真实姓名")
    phone: Optional[str] = Field(None, description="手机号码")
    avatar: Optional[str] = Field(None, description="头像URL")
    is_active: bool = Field(True, description="是否激活")
    is_verified: bool = Field(False, description="是否验证")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    last_login: Optional[float] = Field(None, description="最后登录时间")
    
    @validator('username')
    def username_alphanumeric(cls, v):
        """验证用户名是字母数字下划线组合"""
        if not v.replace('_', '').isalnum():
            raise ValueError('用户名只能包含字母、数字和下划线')
        return v
    
    def update_last_login(self) -> None:
        """更新最后登录时间"""
        self.last_login = time.time()
    
    def to_public_dict(self) -> Dict[str, Any]:
        """转换为公开信息字典（不含敏感数据）"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "real_name": self.real_name,
            "avatar": self.avatar,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "grade_level": self.profile.grade_level if self.profile else None,
            "subjects": self.profile.subjects if self.profile else [],
        } 