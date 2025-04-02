#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
个性化练习推荐引擎

该模块实现了基于学生学习历史和掌握程度的个性化练习题推荐功能。
核心功能包括:
1. 学生模型: 跟踪学生的学习状态和知识掌握程度
2. 遗忘曲线: 基于艾宾浩斯遗忘曲线模型，预测学生的知识遗忘情况
3. 个性化推荐: 根据学生模型和遗忘曲线，推荐合适的练习题
4. 学习路径规划: 根据学习目标，规划最优的学习路径
"""

import os
import time
import logging
import random
import json
import datetime
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Set

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ----- 数据模型 -----

class KnowledgePoint(BaseModel):
    """知识点数据模型"""
    id: str = Field(..., description="知识点ID")
    name: str = Field(..., description="知识点名称")
    subject: str = Field(..., description="所属学科")
    grade_level: int = Field(..., description="适用年级")
    difficulty: float = Field(0.5, description="难度系数(0-1)")
    prerequisites: List[str] = Field(default=[], description="前置知识点ID列表")
    related_points: List[str] = Field(default=[], description="相关知识点ID列表")
    description: str = Field("", description="知识点描述")
    category: str = Field("", description="知识点类别")
    tags: List[str] = Field(default=[], description="标签列表")
    vector: Optional[List[float]] = Field(None, description="知识点向量表示")


class Exercise(BaseModel):
    """练习题数据模型"""
    id: str = Field(..., description="练习题ID")
    title: str = Field(..., description="题目标题")
    content: str = Field(..., description="题目内容")
    answer: str = Field(..., description="标准答案")
    explanation: str = Field("", description="解析")
    knowledge_points: List[str] = Field(..., description="相关知识点ID列表")
    subject: str = Field(..., description="所属学科")
    grade_level: int = Field(..., description="适用年级")
    difficulty: float = Field(0.5, description="难度系数(0-1)")
    type: str = Field(..., description="题目类型(选择题/填空题/解答题等)")
    tags: List[str] = Field(default=[], description="标签列表")
    source: str = Field("", description="来源")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    vector: Optional[List[float]] = Field(None, description="题目向量表示")


class KnowledgeMastery(BaseModel):
    """知识掌握度数据模型"""
    knowledge_point_id: str = Field(..., description="知识点ID")
    mastery_level: float = Field(0.0, description="掌握程度(0-1)")
    last_practice_time: Optional[float] = Field(None, description="上次练习时间")
    practice_count: int = Field(0, description="练习次数")
    correct_count: int = Field(0, description="正确次数")
    wrong_answers: List[str] = Field(default=[], description="错误的题目ID列表")
    retention_rate: float = Field(0.0, description="记忆保持率(0-1)")
    update_time: float = Field(default_factory=time.time, description="更新时间")


class PracticeRecord(BaseModel):
    """练习记录数据模型"""
    id: str = Field(..., description="记录ID")
    student_id: str = Field(..., description="学生ID")
    exercise_id: str = Field(..., description="练习题ID")
    knowledge_points: List[str] = Field(..., description="相关知识点ID列表")
    is_correct: bool = Field(..., description="是否正确")
    answer: str = Field(..., description="学生答案")
    time_spent: float = Field(..., description="耗时(秒)")
    practice_time: float = Field(..., description="练习时间戳")
    difficulty: float = Field(0.5, description="难度系数(0-1)")
    feedback: Optional[str] = Field(None, description="反馈")


class StudentModel(BaseModel):
    """学生模型数据模型"""
    id: str = Field(..., description="学生ID")
    name: str = Field(..., description="学生姓名")
    grade_level: int = Field(..., description="年级")
    knowledge_mastery: Dict[str, KnowledgeMastery] = Field(
        default={}, description="知识掌握情况，键为知识点ID"
    )
    learning_rate: float = Field(0.1, description="学习速率(0-1)")
    forgetting_rate: float = Field(0.05, description="遗忘速率(0-1)")
    practice_preferences: Dict[str, float] = Field(
        default={}, description="练习偏好，键为类型，值为偏好程度(0-1)"
    )
    last_active_time: float = Field(default_factory=time.time, description="上次活跃时间")


class RecommendationRequest(BaseModel):
    """推荐请求数据模型"""
    student_id: str = Field(..., description="学生ID")
    subject: Optional[str] = Field(None, description="学科")
    knowledge_points: Optional[List[str]] = Field(None, description="指定知识点")
    count: int = Field(5, description="推荐数量")
    difficulty_range: Tuple[float, float] = Field((0.0, 1.0), description="难度范围")
    exclude_practiced: bool = Field(False, description="是否排除已练习")
    practice_type: Optional[str] = Field(None, description="练习类型")
    priority: str = Field("balanced", description="优先策略(需要加强/掌握巩固/平衡)")


class RecommendationResponse(BaseModel):
    """推荐响应数据模型"""
    student_id: str = Field(..., description="学生ID")
    recommendations: List[Exercise] = Field(..., description="推荐的练习题列表")
    reasons: List[str] = Field(..., description="推荐理由列表")
    knowledge_status: Dict[str, float] = Field(
        ..., description="相关知识点当前掌握状态，键为知识点ID，值为掌握程度"
    )
    recommendation_time: float = Field(default_factory=time.time, description="推荐时间")


# ----- 遗忘曲线模型 -----

class ForgettingCurveModel:
    """艾宾浩斯遗忘曲线模型"""
    
    def __init__(
        self, 
        base_retention: float = 0.9,
        forgetting_rate: float = 0.1,
        practice_boost: float = 0.1
    ):
        """
        初始化遗忘曲线模型
        
        Args:
            base_retention: 基础记忆保持率
            forgetting_rate: 遗忘速率
            practice_boost: 每次练习提升的记忆强度
        """
        self.base_retention = base_retention
        self.forgetting_rate = forgetting_rate
        self.practice_boost = practice_boost
    
    def calculate_retention(
        self, 
        last_practice_time: float,
        current_time: float,
        practice_count: int,
        learning_strength: float
    ) -> float:
        """
        计算当前记忆保持率
        
        Args:
            last_practice_time: 上次练习时间
            current_time: 当前时间
            practice_count: 练习次数
            learning_strength: 学习强度(0-1)
            
        Returns:
            当前记忆保持率(0-1)
        """
        if last_practice_time is None:
            return 0.0
        
        # 计算时间间隔(天)
        days_passed = (current_time - last_practice_time) / (24 * 3600)
        
        # 计算基础记忆强度
        strength = min(self.base_retention + practice_count * self.practice_boost * learning_strength, 1.0)
        
        # 应用艾宾浩斯遗忘曲线公式
        # R = e^(-k*t), k为遗忘率，t为时间间隔
        retention = strength * math.exp(-self.forgetting_rate * days_passed)
        
        # 确保结果在0-1范围内
        return max(0.0, min(retention, 1.0))
    
    def predict_practice_impact(
        self,
        current_mastery: float,
        is_correct: bool,
        difficulty: float,
        learning_rate: float
    ) -> float:
        """
        预测练习对掌握度的影响
        
        Args:
            current_mastery: 当前掌握度
            is_correct: 是否回答正确
            difficulty: 题目难度
            learning_rate: 学习速率
            
        Returns:
            新的掌握度(0-1)
        """
        if is_correct:
            # 正确回答会提高掌握度，难题提升更多
            gain = learning_rate * difficulty * (1 - current_mastery)
            new_mastery = current_mastery + gain
        else:
            # 错误回答会降低掌握度，简单题降低更多(因为应该会)
            loss = learning_rate * (1 - difficulty) * current_mastery * 0.5
            new_mastery = current_mastery - loss
        
        # 确保结果在0-1范围内
        return max(0.0, min(new_mastery, 1.0))


# ----- 推荐引擎 -----

class RecommendationEngine:
    """个性化练习推荐引擎"""
    
    def __init__(
        self,
        knowledge_db=None,
        exercise_db=None,
        student_db=None,
        practice_db=None,
        forgetting_model=None
    ):
        """
        初始化推荐引擎
        
        Args:
            knowledge_db: 知识点数据库
            exercise_db: 练习题数据库
            student_db: 学生模型数据库
            practice_db: 练习记录数据库
            forgetting_model: 遗忘曲线模型
        """
        self.knowledge_db = knowledge_db
        self.exercise_db = exercise_db
        self.student_db = student_db
        self.practice_db = practice_db
        self.forgetting_model = forgetting_model or ForgettingCurveModel()
        
        logger.info("初始化个性化练习推荐引擎")
    
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        推荐练习题
        
        Args:
            request: 推荐请求
            
        Returns:
            推荐响应，包含推荐的练习题和理由
        """
        logger.info(f"处理推荐请求 - 学生: {request.student_id}, 学科: {request.subject}")
        
        # 1. 获取学生模型
        student = self._get_student_model(request.student_id)
        if not student:
            logger.warning(f"未找到学生模型: {request.student_id}")
            return RecommendationResponse(
                student_id=request.student_id,
                recommendations=[],
                reasons=["未找到学生模型"],
                knowledge_status={}
            )
        
        # 2. 更新学生知识掌握状态(考虑遗忘曲线)
        self._update_knowledge_retention(student)
        
        # 3. 获取候选练习题
        candidates = self._get_candidate_exercises(request, student)
        
        # 4. 对候选题进行评分和排序
        scored_exercises = self._score_exercises(candidates, student, request)
        
        # 5. 选择最终推荐的练习题
        recommendations, reasons = self._select_recommendations(scored_exercises, request.count)
        
        # 6. 获取相关知识点的掌握状态
        knowledge_status = {}
        for exercise in recommendations:
            for kp_id in exercise.knowledge_points:
                if kp_id in student.knowledge_mastery:
                    knowledge_status[kp_id] = student.knowledge_mastery[kp_id].mastery_level
                else:
                    knowledge_status[kp_id] = 0.0
        
        # 7. 构建并返回响应
        return RecommendationResponse(
            student_id=request.student_id,
            recommendations=recommendations,
            reasons=reasons,
            knowledge_status=knowledge_status
        )
    
    def update_student_model(
        self,
        student_id: str,
        practice_record: PracticeRecord
    ) -> StudentModel:
        """
        更新学生模型
        
        Args:
            student_id: 学生ID
            practice_record: 练习记录
            
        Returns:
            更新后的学生模型
        """
        logger.info(f"更新学生模型 - 学生: {student_id}, 练习: {practice_record.exercise_id}")
        
        # 1. 获取学生模型
        student = self._get_student_model(student_id)
        if not student:
            logger.warning(f"未找到学生模型: {student_id}")
            return None
        
        # 2. 获取练习题信息
        exercise = self._get_exercise(practice_record.exercise_id)
        if not exercise:
            logger.warning(f"未找到练习题: {practice_record.exercise_id}")
            return student
        
        # 3. 更新学生的知识掌握度
        for kp_id in exercise.knowledge_points:
            # 如果学生模型中没有该知识点的掌握记录，创建一个新的
            if kp_id not in student.knowledge_mastery:
                student.knowledge_mastery[kp_id] = KnowledgeMastery(
                    knowledge_point_id=kp_id,
                    mastery_level=0.0,
                    last_practice_time=None,
                    practice_count=0,
                    correct_count=0,
                    wrong_answers=[],
                    retention_rate=0.0,
                )
            
            # 获取当前掌握度记录
            mastery = student.knowledge_mastery[kp_id]
            
            # 更新练习次数和正确次数
            mastery.practice_count += 1
            if practice_record.is_correct:
                mastery.correct_count += 1
            else:
                mastery.wrong_answers.append(practice_record.exercise_id)
            
            # 更新掌握程度
            current_mastery = mastery.mastery_level
            new_mastery = self.forgetting_model.predict_practice_impact(
                current_mastery=current_mastery,
                is_correct=practice_record.is_correct,
                difficulty=exercise.difficulty,
                learning_rate=student.learning_rate
            )
            mastery.mastery_level = new_mastery
            
            # 更新最后练习时间
            mastery.last_practice_time = practice_record.practice_time
            
            # 更新记录时间
            mastery.update_time = time.time()
        
        # 4. 更新学生最后活跃时间
        student.last_active_time = time.time()
        
        # 5. 保存更新后的学生模型
        self._save_student_model(student)
        
        logger.info(f"学生模型更新完成 - 学生: {student_id}")
        return student
    
    def _get_student_model(self, student_id: str) -> Optional[StudentModel]:
        """
        获取学生模型
        
        Args:
            student_id: 学生ID
            
        Returns:
            学生模型或None(如果不存在)
        """
        # 实际项目中，这里会从数据库获取学生模型
        # 这里模拟获取一个示例学生模型
        
        if self.student_db is not None:
            # 从实际数据库获取
            return self.student_db.get(student_id)
        
        # 模拟数据
        mock_student = StudentModel(
            id=student_id,
            name=f"学生{student_id}",
            grade_level=6,  # 六年级
            knowledge_mastery={
                "kp_math_01": KnowledgeMastery(
                    knowledge_point_id="kp_math_01",
                    mastery_level=0.8,
                    last_practice_time=time.time() - 7 * 24 * 3600,  # 7天前
                    practice_count=10,
                    correct_count=8,
                    retention_rate=0.7
                ),
                "kp_math_02": KnowledgeMastery(
                    knowledge_point_id="kp_math_02",
                    mastery_level=0.6,
                    last_practice_time=time.time() - 3 * 24 * 3600,  # 3天前
                    practice_count=5,
                    correct_count=3,
                    retention_rate=0.6
                ),
                "kp_math_03": KnowledgeMastery(
                    knowledge_point_id="kp_math_03",
                    mastery_level=0.3,
                    last_practice_time=time.time() - 10 * 24 * 3600,  # 10天前
                    practice_count=2,
                    correct_count=0,
                    retention_rate=0.2
                ),
            },
            learning_rate=0.15,
            forgetting_rate=0.05,
            practice_preferences={
                "选择题": 0.8,
                "填空题": 0.6,
                "解答题": 0.4
            }
        )
        return mock_student
    
    def _save_student_model(self, student: StudentModel) -> bool:
        """
        保存学生模型
        
        Args:
            student: 学生模型
            
        Returns:
            是否保存成功
        """
        # 实际项目中，这里会将学生模型保存到数据库
        # 这里只记录日志
        logger.info(f"保存学生模型: {student.id}")
        
        if self.student_db is not None:
            # 保存到实际数据库
            return self.student_db.save(student)
        
        return True
    
    def _update_knowledge_retention(self, student: StudentModel) -> None:
        """
        更新学生的知识保持率(考虑遗忘曲线)
        
        Args:
            student: 学生模型
        """
        current_time = time.time()
        
        for kp_id, mastery in student.knowledge_mastery.items():
            # 计算当前的记忆保持率
            retention = self.forgetting_model.calculate_retention(
                last_practice_time=mastery.last_practice_time,
                current_time=current_time,
                practice_count=mastery.practice_count,
                learning_strength=student.learning_rate
            )
            
            # 更新保持率
            mastery.retention_rate = retention
            
            # 更新当前掌握度(考虑遗忘)
            # 掌握度 = 原始掌握度 × 保持率
            if mastery.practice_count > 0:  # 只有练习过的知识点才会遗忘
                original_mastery = mastery.mastery_level
                new_mastery = original_mastery * retention
                mastery.mastery_level = new_mastery
    
    def _get_candidate_exercises(
        self,
        request: RecommendationRequest,
        student: StudentModel
    ) -> List[Exercise]:
        """
        获取候选练习题列表
        
        Args:
            request: 推荐请求
            student: 学生模型
            
        Returns:
            候选练习题列表
        """
        # 实际项目中，这里会根据条件从数据库查询练习题
        # 这里返回模拟数据
        
        if self.exercise_db is not None:
            # 从实际数据库查询
            query = {}
            
            if request.subject:
                query["subject"] = request.subject
            
            if request.knowledge_points:
                query["knowledge_points"] = {"$in": request.knowledge_points}
            
            if request.practice_type:
                query["type"] = request.practice_type
            
            min_diff, max_diff = request.difficulty_range
            query["difficulty"] = {"$gte": min_diff, "$lte": max_diff}
            
            query["grade_level"] = {"$lte": student.grade_level}
            
            # 排除已练习的题
            if request.exclude_practiced:
                practiced_exercises = self._get_practiced_exercises(student.id)
                if practiced_exercises:
                    query["id"] = {"$nin": practiced_exercises}
            
            return self.exercise_db.find(query)
        
        # 模拟数据
        mock_exercises = [
            Exercise(
                id="ex_math_01",
                title="分数加法",
                content="计算: 1/2 + 1/3 = ?",
                answer="5/6",
                explanation="通分后相加: 3/6 + 2/6 = 5/6",
                knowledge_points=["kp_math_01", "kp_math_02"],
                subject="数学",
                grade_level=5,
                difficulty=0.4,
                type="填空题",
                tags=["分数", "加法"]
            ),
            Exercise(
                id="ex_math_02",
                title="分数减法",
                content="计算: 3/4 - 1/6 = ?",
                answer="7/12",
                explanation="通分后相减: 9/12 - 2/12 = 7/12",
                knowledge_points=["kp_math_01", "kp_math_02"],
                subject="数学",
                grade_level=5,
                difficulty=0.5,
                type="填空题",
                tags=["分数", "减法"]
            ),
            Exercise(
                id="ex_math_03",
                title="小数乘法",
                content="计算: 0.25 × 0.4 = ?",
                answer="0.1",
                explanation="0.25 × 0.4 = 25/100 × 4/10 = 100/1000 = 0.1",
                knowledge_points=["kp_math_03"],
                subject="数学",
                grade_level=6,
                difficulty=0.6,
                type="填空题",
                tags=["小数", "乘法"]
            ),
            Exercise(
                id="ex_math_04",
                title="比例应用",
                content="小明家到学校的距离是3千米，地图上的距离是6厘米，求地图的比例尺。",
                answer="1:50000",
                explanation="3千米=300000厘米，比例尺=6:300000=1:50000",
                knowledge_points=["kp_math_04"],
                subject="数学",
                grade_level=6,
                difficulty=0.7,
                type="解答题",
                tags=["比例", "应用题"]
            ),
            Exercise(
                id="ex_math_05",
                title="方程解答",
                content="解方程: 2x + 5 = 3x - 4",
                answer="x = 9",
                explanation="2x + 5 = 3x - 4, 移项得 5 + 4 = 3x - 2x, 9 = x, 所以 x = 9",
                knowledge_points=["kp_math_05"],
                subject="数学",
                grade_level=6,
                difficulty=0.6,
                type="解答题",
                tags=["方程", "解方程"]
            ),
            Exercise(
                id="ex_math_06",
                title="面积计算",
                content="一个长方形的长是8厘米，宽是5厘米，求它的面积。",
                answer="40平方厘米",
                explanation="长方形面积=长×宽=8×5=40(平方厘米)",
                knowledge_points=["kp_math_06"],
                subject="数学",
                grade_level=4,
                difficulty=0.3,
                type="填空题",
                tags=["几何", "面积"]
            ),
        ]
        
        # 根据请求条件过滤
        filtered = mock_exercises
        
        if request.subject:
            filtered = [e for e in filtered if e.subject == request.subject]
        
        if request.knowledge_points:
            filtered = [
                e for e in filtered
                if any(kp in request.knowledge_points for kp in e.knowledge_points)
            ]
        
        if request.practice_type:
            filtered = [e for e in filtered if e.type == request.practice_type]
        
        min_diff, max_diff = request.difficulty_range
        filtered = [
            e for e in filtered
            if min_diff <= e.difficulty <= max_diff
        ]
        
        # 按年级过滤
        filtered = [e for e in filtered if e.grade_level <= student.grade_level]
        
        # 排除已练习的题
        if request.exclude_practiced:
            practiced_exercises = self._get_practiced_exercises(student.id)
            if practiced_exercises:
                filtered = [e for e in filtered if e.id not in practiced_exercises]
        
        return filtered
    
    def _get_practiced_exercises(self, student_id: str) -> Set[str]:
        """
        获取学生已练习过的题目ID集合
        
        Args:
            student_id: 学生ID
            
        Returns:
            已练习题目ID集合
        """
        # 实际项目中，这里会从数据库查询学生的练习记录
        # 这里返回一个空集合
        
        if self.practice_db is not None:
            # 从实际数据库查询
            records = self.practice_db.find({"student_id": student_id})
            return {record.exercise_id for record in records}
        
        # 模拟数据
        return set()
    
    def _get_exercise(self, exercise_id: str) -> Optional[Exercise]:
        """
        获取练习题详情
        
        Args:
            exercise_id: 练习题ID
            
        Returns:
            练习题或None(如果不存在)
        """
        # 实际项目中，这里会从数据库获取练习题
        # 这里返回模拟数据
        
        if self.exercise_db is not None:
            # 从实际数据库获取
            return self.exercise_db.get(exercise_id)
        
        # 模拟数据
        mock_exercises = {
            "ex_math_01": Exercise(
                id="ex_math_01",
                title="分数加法",
                content="计算: 1/2 + 1/3 = ?",
                answer="5/6",
                explanation="通分后相加: 3/6 + 2/6 = 5/6",
                knowledge_points=["kp_math_01", "kp_math_02"],
                subject="数学",
                grade_level=5,
                difficulty=0.4,
                type="填空题",
                tags=["分数", "加法"]
            ),
            "ex_math_02": Exercise(
                id="ex_math_02",
                title="分数减法",
                content="计算: 3/4 - 1/6 = ?",
                answer="7/12",
                explanation="通分后相减: 9/12 - 2/12 = 7/12",
                knowledge_points=["kp_math_01", "kp_math_02"],
                subject="数学",
                grade_level=5,
                difficulty=0.5,
                type="填空题",
                tags=["分数", "减法"]
            ),
        }
        
        return mock_exercises.get(exercise_id)
    
    def _score_exercises(
        self,
        exercises: List[Exercise],
        student: StudentModel,
        request: RecommendationRequest
    ) -> List[Tuple[Exercise, float, str]]:
        """
        对候选练习题评分
        
        Args:
            exercises: 候选练习题列表
            student: 学生模型
            request: 推荐请求
            
        Returns:
            (练习题, 分数, 推荐理由)元组的列表，按分数降序排序
        """
        if not exercises:
            return []
        
        scored_exercises = []
        
        for exercise in exercises:
            score = 0.0
            reasons = []
            
            # 1. 知识点掌握度得分
            kp_score = 0.0
            kp_count = 0
            need_practice_kps = []
            
            for kp_id in exercise.knowledge_points:
                if kp_id in student.knowledge_mastery:
                    mastery = student.knowledge_mastery[kp_id]
                    kp_score += mastery.mastery_level
                    kp_count += 1
                    
                    # 记录需要加强的知识点
                    if mastery.mastery_level < 0.6:
                        need_practice_kps.append(kp_id)
                else:
                    # 未练习过的知识点
                    kp_score += 0.0
                    kp_count += 1
                    need_practice_kps.append(kp_id)
            
            if kp_count > 0:
                avg_mastery = kp_score / kp_count
                
                # 根据请求的优先策略调整分数
                if request.priority == "balanced":
                    # 平衡策略：中等掌握度(0.4-0.6)的知识点优先
                    if 0.4 <= avg_mastery <= 0.6:
                        knowledge_score = 1.0
                        reasons.append("包含掌握程度适中的知识点，适合巩固")
                    elif avg_mastery < 0.4:
                        knowledge_score = 0.8
                        reasons.append("包含掌握程度较低的知识点，需要加强")
                    else:  # > 0.6
                        knowledge_score = 0.6
                        reasons.append("包含已较好掌握的知识点，可以复习")
                elif request.priority == "weak_first":
                    # 弱点优先：掌握度低的优先
                    knowledge_score = 1.0 - avg_mastery
                    if avg_mastery < 0.4:
                        reasons.append("包含掌握程度较低的知识点，需要重点加强")
                    else:
                        reasons.append("有助于提高薄弱知识点")
                else:  # "reinforce"
                    # 巩固优先：掌握度高的优先
                    knowledge_score = avg_mastery
                    if avg_mastery > 0.6:
                        reasons.append("包含已较好掌握的知识点，适合巩固记忆")
                    else:
                        reasons.append("有助于巩固已学知识")
            else:
                knowledge_score = 0.5
                reasons.append("新知识点，扩展知识面")
            
            # 2. 难度匹配得分
            ideal_difficulty = 0.5
            if kp_count > 0 and avg_mastery > 0:
                # 根据掌握度调整理想难度
                # 掌握度低，难度应低；掌握度高，难度可以高一些
                ideal_difficulty = 0.3 + avg_mastery * 0.5
            
            difficulty_score = 1.0 - min(abs(exercise.difficulty - ideal_difficulty) * 2, 1.0)
            
            if abs(exercise.difficulty - ideal_difficulty) < 0.1:
                reasons.append("难度适中，符合当前水平")
            elif exercise.difficulty < ideal_difficulty:
                reasons.append("难度较低，有助于建立信心")
            else:
                reasons.append("难度较高，有助于提升能力")
            
            # 3. 练习类型偏好得分
            preference_score = 0.5  # 默认中等偏好
            if exercise.type in student.practice_preferences:
                preference_score = student.practice_preferences[exercise.type]
                if preference_score > 0.7:
                    reasons.append(f"包含偏好的题型: {exercise.type}")
            
            # 综合评分 (加权求和)
            # 知识点掌握度权重最高
            final_score = (
                knowledge_score * 0.6 +
                difficulty_score * 0.3 +
                preference_score * 0.1
            )
            
            # 选择最主要的理由
            main_reason = reasons[0] if reasons else "综合练习"
            
            scored_exercises.append((exercise, final_score, main_reason))
        
        # 按分数降序排序
        return sorted(scored_exercises, key=lambda x: x[1], reverse=True)
    
    def _select_recommendations(
        self,
        scored_exercises: List[Tuple[Exercise, float, str]],
        count: int
    ) -> Tuple[List[Exercise], List[str]]:
        """
        从评分后的练习题中选择最终推荐的题目
        
        Args:
            scored_exercises: (练习题, 分数, 理由)元组的列表
            count: 需要推荐的数量
            
        Returns:
            (推荐的练习题列表, 推荐理由列表)
        """
        if not scored_exercises:
            return [], []
        
        # 简单截取前count个
        selected = scored_exercises[:count]
        
        # 提取练习题和理由
        exercises = [item[0] for item in selected]
        reasons = [item[2] for item in selected]
        
        return exercises, reasons


# 创建推荐引擎实例
def create_recommendation_engine(
    knowledge_db=None,
    exercise_db=None,
    student_db=None,
    practice_db=None,
    forgetting_model=None
) -> RecommendationEngine:
    """
    创建个性化练习推荐引擎实例
    
    Args:
        knowledge_db: 知识点数据库
        exercise_db: 练习题数据库
        student_db: 学生模型数据库
        practice_db: 练习记录数据库
        forgetting_model: 遗忘曲线模型
        
    Returns:
        推荐引擎实例
    """
    return RecommendationEngine(
        knowledge_db=knowledge_db,
        exercise_db=exercise_db,
        student_db=student_db,
        practice_db=practice_db,
        forgetting_model=forgetting_model or ForgettingCurveModel()
    ) 