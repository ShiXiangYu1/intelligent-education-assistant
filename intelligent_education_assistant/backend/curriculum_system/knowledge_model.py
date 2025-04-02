#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
课标知识点数据模型

定义了系统中知识点相关的数据模型，包括学科、年级、知识点、关系类型等。
"""

import time
import enum
import uuid
from typing import Dict, List, Optional, Set, Union, Any
from pydantic import BaseModel, Field, validator


class Subject(str, enum.Enum):
    """学科枚举"""
    CHINESE = "chinese"         # 语文
    MATH = "math"               # 数学
    ENGLISH = "english"         # 英语
    PHYSICS = "physics"         # 物理
    CHEMISTRY = "chemistry"     # 化学
    BIOLOGY = "biology"         # 生物
    HISTORY = "history"         # 历史
    GEOGRAPHY = "geography"     # 地理
    POLITICS = "politics"       # 政治
    GENERAL = "general"         # 通用


class GradeLevel(str, enum.Enum):
    """年级枚举"""
    PRIMARY_1 = "primary_1"  # 小学一年级
    PRIMARY_2 = "primary_2"  # 小学二年级
    PRIMARY_3 = "primary_3"  # 小学三年级
    PRIMARY_4 = "primary_4"  # 小学四年级
    PRIMARY_5 = "primary_5"  # 小学五年级
    PRIMARY_6 = "primary_6"  # 小学六年级
    JUNIOR_1 = "junior_1"    # 初中一年级
    JUNIOR_2 = "junior_2"    # 初中二年级
    JUNIOR_3 = "junior_3"    # 初中三年级
    SENIOR_1 = "senior_1"    # 高中一年级
    SENIOR_2 = "senior_2"    # 高中二年级
    SENIOR_3 = "senior_3"    # 高中三年级
    
    @classmethod
    def to_numeric(cls, grade: 'GradeLevel') -> int:
        """将枚举值转换为数字表示（1-12）"""
        mapping = {
            cls.PRIMARY_1: 1,
            cls.PRIMARY_2: 2,
            cls.PRIMARY_3: 3,
            cls.PRIMARY_4: 4,
            cls.PRIMARY_5: 5,
            cls.PRIMARY_6: 6,
            cls.JUNIOR_1: 7,
            cls.JUNIOR_2: 8,
            cls.JUNIOR_3: 9,
            cls.SENIOR_1: 10,
            cls.SENIOR_2: 11,
            cls.SENIOR_3: 12,
        }
        return mapping.get(grade, 0)
    
    @classmethod
    def from_numeric(cls, num: int) -> Optional['GradeLevel']:
        """从数字表示（1-12）转换为枚举值"""
        mapping = {
            1: cls.PRIMARY_1,
            2: cls.PRIMARY_2,
            3: cls.PRIMARY_3,
            4: cls.PRIMARY_4,
            5: cls.PRIMARY_5,
            6: cls.PRIMARY_6,
            7: cls.JUNIOR_1,
            8: cls.JUNIOR_2,
            9: cls.JUNIOR_3,
            10: cls.SENIOR_1,
            11: cls.SENIOR_2,
            12: cls.SENIOR_3,
        }
        return mapping.get(num, None)
    
    @classmethod
    def get_stage(cls, grade: 'GradeLevel') -> str:
        """获取学段信息"""
        if grade in [cls.PRIMARY_1, cls.PRIMARY_2, cls.PRIMARY_3, cls.PRIMARY_4, cls.PRIMARY_5, cls.PRIMARY_6]:
            return "小学"
        elif grade in [cls.JUNIOR_1, cls.JUNIOR_2, cls.JUNIOR_3]:
            return "初中"
        elif grade in [cls.SENIOR_1, cls.SENIOR_2, cls.SENIOR_3]:
            return "高中"
        return "未知"


class RelationType(str, enum.Enum):
    """知识点关系类型枚举"""
    PREREQUISITE = "prerequisite"      # 前置知识
    RELATED = "related"                # 相关知识
    EXTENSION = "extension"            # 拓展知识
    INCLUDES = "includes"              # 包含关系
    BELONGS_TO = "belongs_to"          # 从属关系
    FOLLOWED_BY = "followed_by"        # 后继知识


class KnowledgePoint(BaseModel):
    """知识点模型"""
    id: str = Field(..., description="知识点ID")
    name: str = Field(..., description="知识点名称")
    subject: Subject = Field(..., description="所属学科")
    grade_level: GradeLevel = Field(..., description="适用年级")
    description: str = Field("", description="知识点描述")
    keywords: List[str] = Field(default=[], description="关键词")
    difficulty: float = Field(0.5, description="难度系数(0-1)")
    importance: float = Field(0.5, description="重要性(0-1)")
    parent_id: Optional[str] = Field(None, description="父级知识点ID")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    updated_at: float = Field(default_factory=time.time, description="更新时间")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    
    @validator('difficulty', 'importance')
    def check_range(cls, v, field):
        """检查数值范围"""
        if v < 0 or v > 1:
            field_name = field.name
            raise ValueError(f"{field_name}必须在0到1之间")
        return v
    
    @staticmethod
    def create_id(subject: Subject, name: str) -> str:
        """
        创建知识点ID
        
        Args:
            subject: 学科
            name: 知识点名称
            
        Returns:
            知识点ID
        """
        base = f"{subject}_{name}_{uuid.uuid4().hex[:8]}"
        return base.lower().replace(" ", "_")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "subject": self.subject,
            "grade_level": self.grade_level,
            "description": self.description,
            "keywords": self.keywords,
            "difficulty": self.difficulty,
            "importance": self.importance,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


class KnowledgeRelation(BaseModel):
    """知识点关系模型"""
    source_id: str = Field(..., description="源知识点ID")
    target_id: str = Field(..., description="目标知识点ID")
    relation_type: RelationType = Field(..., description="关系类型")
    weight: float = Field(1.0, description="关系权重")
    description: str = Field("", description="关系描述")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    
    def get_key(self) -> str:
        """获取关系键"""
        return f"{self.source_id}_{self.target_id}_{self.relation_type}"


class Curriculum(BaseModel):
    """课程体系模型"""
    id: str = Field(..., description="课程ID")
    name: str = Field(..., description="课程名称")
    subject: Subject = Field(..., description="所属学科")
    grade_level: GradeLevel = Field(..., description="适用年级")
    description: str = Field("", description="课程描述")
    knowledge_points: List[str] = Field(default=[], description="包含的知识点ID列表")
    prerequisites: List[str] = Field(default=[], description="先修课程ID列表")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    updated_at: float = Field(default_factory=time.time, description="更新时间")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    
    @staticmethod
    def create_id(subject: Subject, name: str, grade_level: GradeLevel) -> str:
        """
        创建课程ID
        
        Args:
            subject: 学科
            name: 课程名称
            grade_level: 年级
            
        Returns:
            课程ID
        """
        base = f"{subject}_{grade_level}_{name}_{uuid.uuid4().hex[:8]}"
        return base.lower().replace(" ", "_") 