#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用户学习整合模块

该模块提供用户服务模块和推荐引擎模块之间的数据整合功能，
实现用户画像与学习记录的同步、分析和可视化。
"""

import time
import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

# 导入用户服务模块
from .user_model import User, UserProfile, LearningRecord, UserRole
from .user_service import UserService

# 导入推荐引擎模块
from backend.recommendation_engine import (
    RecommendationEngine, StudentModel, KnowledgeMastery, 
    PracticeRecord, Exercise
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LearningStats(BaseModel):
    """学习统计数据模型"""
    total_exercises: int = Field(0, description="总练习题数")
    correct_exercises: int = Field(0, description="正确题目数")
    accuracy_rate: float = Field(0.0, description="准确率")
    total_time: float = Field(0.0, description="总学习时间(分钟)")
    average_time_per_exercise: float = Field(0.0, description="平均每题用时(分钟)")
    subjects_distribution: Dict[str, int] = Field(default={}, description="学科分布")
    knowledge_mastery: Dict[str, float] = Field(default={}, description="知识点掌握度")
    daily_activity: Dict[str, int] = Field(default={}, description="每日活动量")
    learning_trend: Dict[str, float] = Field(default={}, description="学习趋势")
    strong_points: List[str] = Field(default=[], description="擅长知识点")
    weak_points: List[str] = Field(default=[], description="薄弱知识点")
    recommended_focus: List[str] = Field(default=[], description="推荐关注知识点")


class UserLearningConfig(BaseModel):
    """用户学习整合配置"""
    sync_interval: int = Field(3600, description="同步间隔(秒)")
    strong_threshold: float = Field(0.8, description="擅长阈值")
    weak_threshold: float = Field(0.4, description="薄弱阈值")
    max_strong_points: int = Field(10, description="最大擅长知识点数量")
    max_weak_points: int = Field(10, description="最大薄弱知识点数量")
    max_recommended_focus: int = Field(5, description="最大推荐关注知识点数量")
    learning_trend_days: int = Field(30, description="学习趋势统计天数")
    learning_history_max_days: int = Field(365, description="学习历史最大保存天数")
    enable_auto_sync: bool = Field(True, description="启用自动同步")
    enable_analytics: bool = Field(True, description="启用学习分析")


class UserLearningIntegration:
    """用户学习整合类，处理用户画像与学习记录整合"""
    
    def __init__(
        self,
        user_service: UserService,
        recommendation_engine: RecommendationEngine,
        config: UserLearningConfig = None
    ):
        """
        初始化用户学习整合
        
        Args:
            user_service: 用户服务
            recommendation_engine: 推荐引擎
            config: 配置
        """
        self.user_service = user_service
        self.recommendation_engine = recommendation_engine
        self.config = config or UserLearningConfig()
        self.last_sync_time = {}  # 用户ID -> 上次同步时间
        self._analytics_cache = {}  # 用户ID -> 分析结果缓存
        self._learning_history = {}  # 用户ID -> {日期 -> 学习记录}
        
        logger.info("初始化用户学习整合模块")
        
        # 启动自动同步任务
        if self.config.enable_auto_sync:
            self._start_auto_sync()
    
    def sync_user_to_student_model(self, user_id: str) -> Tuple[bool, str]:
        """
        将用户画像同步到学生模型
        
        Args:
            user_id: 用户ID
            
        Returns:
            同步结果和消息
        """
        logger.info(f"同步用户画像到学生模型 - 用户ID: {user_id}")
        
        # 获取用户
        user = self.user_service.find_by_id(user_id)
        if not user:
            return False, f"用户不存在: {user_id}"
        
        # 只同步学生角色
        if user.role != UserRole.STUDENT:
            return False, f"用户不是学生角色: {user.role}"
        
        # 构建学生模型
        student_model = self._build_student_model_from_user(user)
        
        # 更新到推荐引擎
        if self.recommendation_engine.student_db:
            self.recommendation_engine.student_db[user_id] = student_model
            self.last_sync_time[user_id] = time.time()
            return True, f"用户 {user_id} 同步到学生模型成功"
        else:
            return False, "推荐引擎未配置学生数据库"
    
    def sync_learning_records_to_user(self, user_id: str) -> Tuple[bool, str]:
        """
        将学习记录同步到用户画像
        
        Args:
            user_id: 用户ID
            
        Returns:
            同步结果和消息
        """
        logger.info(f"同步学习记录到用户画像 - 用户ID: {user_id}")
        
        # 获取用户
        user = self.user_service.find_by_id(user_id)
        if not user:
            return False, f"用户不存在: {user_id}"
        
        # 只同步学生角色
        if user.role != UserRole.STUDENT:
            return False, f"用户不是学生角色: {user.role}"
        
        # 获取学生模型
        student_model = None
        if self.recommendation_engine.student_db:
            student_model = self.recommendation_engine.student_db.get(user_id)
        
        if not student_model:
            return False, f"学生模型不存在: {user_id}"
        
        # 更新用户画像
        self._update_user_profile_from_student(user, student_model)
        
        # 保存用户
        self.user_service.update(user)
        self.last_sync_time[user_id] = time.time()
        
        return True, f"用户 {user_id} 学习记录同步成功"
    
    def process_practice_record(
        self, 
        user_id: str, 
        exercise_id: str,
        is_correct: bool,
        answer: str,
        time_spent: float,
        knowledge_points: List[str] = None
    ) -> Tuple[bool, str]:
        """
        处理练习记录，同时更新用户画像和学生模型
        
        Args:
            user_id: 用户ID
            exercise_id: 练习题ID
            is_correct: 是否正确
            answer: 学生答案
            time_spent: 耗时(秒)
            knowledge_points: 相关知识点ID列表
            
        Returns:
            处理结果和消息
        """
        logger.info(f"处理练习记录 - 用户: {user_id}, 练习题: {exercise_id}")
        
        # 获取用户
        user = self.user_service.find_by_id(user_id)
        if not user:
            return False, f"用户不存在: {user_id}"
        
        # 获取练习题
        exercise = self.recommendation_engine._get_exercise(exercise_id)
        if not exercise:
            return False, f"练习题不存在: {exercise_id}"
        
        # 如果未提供知识点，使用练习题的知识点
        if knowledge_points is None and exercise:
            knowledge_points = exercise.knowledge_points
        
        # 构建练习记录
        record_id = f"{user_id}_{exercise_id}_{int(time.time())}"
        practice_record = PracticeRecord(
            id=record_id,
            student_id=user_id,
            exercise_id=exercise_id,
            knowledge_points=knowledge_points or [],
            is_correct=is_correct,
            answer=answer,
            time_spent=time_spent,
            practice_time=time.time(),
            difficulty=exercise.difficulty if exercise else 0.5
        )
        
        # 更新学生模型
        student_model = self.recommendation_engine.update_student_model(
            user_id, practice_record
        )
        
        if not student_model:
            # 如果学生模型不存在，创建一个新的并从用户同步
            success, message = self.sync_user_to_student_model(user_id)
            if success:
                # 再次尝试更新学生模型
                student_model = self.recommendation_engine.update_student_model(
                    user_id, practice_record
                )
        
        # 更新用户画像中的学习记录
        if student_model:
            # 从更新后的学生模型获取知识点掌握情况
            knowledge_mastery = {}
            for kp_id in knowledge_points or []:
                if kp_id in student_model.knowledge_mastery:
                    knowledge_mastery[kp_id] = student_model.knowledge_mastery[kp_id].mastery_level
            
            # 更新用户画像
            user.profile.update_learning_record(
                subject=exercise.subject if exercise else "general",
                is_correct=is_correct,
                time_spent=time_spent/60.0,  # 转换为分钟
                knowledge_points=knowledge_mastery
            )
            
            # 更新学习历史
            self._update_learning_history(user_id, exercise, is_correct, time_spent)
            
            # 保存用户
            self.user_service.update(user)
            self.last_sync_time[user_id] = time.time()
            
            # 清除分析缓存
            if user_id in self._analytics_cache:
                del self._analytics_cache[user_id]
            
            return True, f"用户 {user_id} 练习记录处理成功"
        else:
            return False, f"用户 {user_id} 学生模型更新失败"
    
    def get_learning_stats(self, user_id: str, refresh: bool = False) -> Optional[LearningStats]:
        """
        获取用户学习统计数据
        
        Args:
            user_id: 用户ID
            refresh: 是否刷新缓存
            
        Returns:
            学习统计数据
        """
        logger.info(f"获取用户学习统计 - 用户ID: {user_id}")
        
        # 如果缓存中有且不需要刷新，直接返回
        if not refresh and user_id in self._analytics_cache:
            return self._analytics_cache[user_id]
        
        # 获取用户
        user = self.user_service.find_by_id(user_id)
        if not user:
            return None
        
        # 获取学生模型
        student_model = None
        if self.recommendation_engine.student_db:
            student_model = self.recommendation_engine.student_db.get(user_id)
        
        # 生成学习统计
        stats = self._generate_learning_stats(user, student_model)
        
        # 缓存结果
        self._analytics_cache[user_id] = stats
        
        return stats
    
    def get_knowledge_mastery_detail(
        self, 
        user_id: str, 
        subject: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        获取用户知识点掌握详情
        
        Args:
            user_id: 用户ID
            subject: 学科(可选)
            
        Returns:
            知识点掌握详情
        """
        logger.info(f"获取知识点掌握详情 - 用户: {user_id}, 学科: {subject}")
        
        # 获取学生模型
        student_model = None
        if self.recommendation_engine.student_db:
            student_model = self.recommendation_engine.student_db.get(user_id)
        
        if not student_model:
            return {}
        
        # 获取知识点掌握情况
        result = {}
        for kp_id, mastery in student_model.knowledge_mastery.items():
            # 如果指定了学科，过滤掉不匹配的
            if subject and not kp_id.startswith(f"kp_{subject}"):
                continue
            
            # 获取知识点详情
            kp_info = self._get_knowledge_point_info(kp_id)
            
            result[kp_id] = {
                "id": kp_id,
                "name": kp_info.get("name", kp_id),
                "subject": kp_info.get("subject", "unknown"),
                "grade_level": kp_info.get("grade_level", 0),
                "mastery_level": mastery.mastery_level,
                "retention_rate": mastery.retention_rate,
                "practice_count": mastery.practice_count,
                "correct_count": mastery.correct_count,
                "last_practice_time": mastery.last_practice_time,
                "accuracy_rate": mastery.correct_count / max(1, mastery.practice_count),
                "description": kp_info.get("description", ""),
                "prerequisites": kp_info.get("prerequisites", []),
                "related_points": kp_info.get("related_points", [])
            }
        
        return result
    
    def get_learning_history(
        self, 
        user_id: str, 
        days: int = 30
    ) -> Dict[str, Dict[str, Any]]:
        """
        获取用户学习历史
        
        Args:
            user_id: 用户ID
            days: 天数
            
        Returns:
            学习历史
        """
        logger.info(f"获取学习历史 - 用户: {user_id}, 天数: {days}")
        
        if user_id not in self._learning_history:
            self._load_learning_history(user_id)
        
        # 获取指定天数的历史
        now = datetime.now()
        start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        
        result = {}
        for date, records in self._learning_history.get(user_id, {}).items():
            if date >= start_date:
                result[date] = records
        
        return result
    
    def get_recommended_learning_path(
        self, 
        user_id: str, 
        subject: str, 
        target_knowledge_point: str = None
    ) -> List[Dict[str, Any]]:
        """
        获取推荐学习路径
        
        Args:
            user_id: 用户ID
            subject: 学科
            target_knowledge_point: 目标知识点(可选)
            
        Returns:
            推荐学习路径
        """
        logger.info(f"获取推荐学习路径 - 用户: {user_id}, 学科: {subject}")
        
        # 获取用户
        user = self.user_service.find_by_id(user_id)
        if not user:
            return []
        
        # 获取学生模型
        student_model = None
        if self.recommendation_engine.student_db:
            student_model = self.recommendation_engine.student_db.get(user_id)
        
        if not student_model:
            return []
        
        # 分析知识点掌握情况
        mastered_kps = []
        weak_kps = []
        for kp_id, mastery in student_model.knowledge_mastery.items():
            # 过滤学科
            if not kp_id.startswith(f"kp_{subject}"):
                continue
                
            if mastery.mastery_level >= self.config.strong_threshold:
                mastered_kps.append(kp_id)
            elif mastery.mastery_level <= self.config.weak_threshold:
                weak_kps.append(kp_id)
        
        # 如果指定了目标知识点，规划学习路径
        if target_knowledge_point:
            return self._plan_learning_path(target_knowledge_point, mastered_kps)
        
        # 否则，推荐基于薄弱点的学习路径
        paths = []
        for weak_kp in weak_kps[:self.config.max_recommended_focus]:
            path = self._plan_learning_path(weak_kp, mastered_kps)
            if path:
                paths.append({
                    "target": weak_kp,
                    "path": path
                })
        
        return paths
    
    def _build_student_model_from_user(self, user: User) -> StudentModel:
        """从用户构建学生模型"""
        # 创建知识掌握情况字典
        knowledge_mastery = {}
        for subject, record in user.profile.learning_records.items():
            for kp_id, mastery_level in record.knowledge_points.items():
                # 创建知识掌握度对象
                knowledge_mastery[kp_id] = KnowledgeMastery(
                    knowledge_point_id=kp_id,
                    mastery_level=mastery_level,
                    last_practice_time=record.last_active,
                    practice_count=record.exercise_count,
                    correct_count=record.correct_count,
                    retention_rate=mastery_level,  # 初始保持率与掌握度相同
                )
        
        # 创建学生模型
        return StudentModel(
            id=user.id,
            name=user.real_name or user.username,
            grade_level=user.profile.grade_level or 1,
            knowledge_mastery=knowledge_mastery,
            learning_rate=user.profile.learning_rate,
            forgetting_rate=user.profile.forgetting_rate,
            last_active_time=max([r.last_active for r in user.profile.learning_records.values()]) if user.profile.learning_records else time.time()
        )
    
    def _update_user_profile_from_student(self, user: User, student: StudentModel) -> None:
        """从学生模型更新用户画像"""
        # 更新基本信息
        user.profile.grade_level = student.grade_level
        user.profile.learning_rate = student.learning_rate
        user.profile.forgetting_rate = student.forgetting_rate
        
        # 整理知识点掌握情况，按学科分组
        subject_knowledge_points = {}
        for kp_id, mastery in student.knowledge_mastery.items():
            # 从知识点ID提取学科
            parts = kp_id.split("_")
            subject = parts[1] if len(parts) > 1 else "general"
            
            if subject not in subject_knowledge_points:
                subject_knowledge_points[subject] = {}
            
            subject_knowledge_points[subject][kp_id] = mastery.mastery_level
        
        # 更新每个学科的学习记录
        for subject, knowledge_points in subject_knowledge_points.items():
            if subject not in user.profile.learning_records:
                user.profile.learning_records[subject] = LearningRecord()
            
            # 更新知识点掌握情况
            user.profile.learning_records[subject].knowledge_points.update(knowledge_points)
        
        # 更新擅长和薄弱知识点
        strong_points = []
        weak_points = []
        
        for kp_id, mastery in student.knowledge_mastery.items():
            if mastery.mastery_level >= self.config.strong_threshold:
                strong_points.append(kp_id)
            elif mastery.mastery_level <= self.config.weak_threshold:
                weak_points.append(kp_id)
        
        user.profile.strong_points = strong_points[:self.config.max_strong_points]
        user.profile.weak_points = weak_points[:self.config.max_weak_points]
        
        # 更新关注的学科
        subjects = set(subject_knowledge_points.keys())
        if subjects:
            user.profile.subjects = list(subjects)
    
    def _generate_learning_stats(
        self, 
        user: User, 
        student_model: Optional[StudentModel]
    ) -> LearningStats:
        """生成学习统计数据"""
        # 基础统计
        total_exercises = 0
        correct_exercises = 0
        total_time = 0.0
        subjects_distribution = {}
        
        # 从用户画像获取基础统计
        for subject, record in user.profile.learning_records.items():
            total_exercises += record.exercise_count
            correct_exercises += record.correct_count
            total_time += record.total_time
            subjects_distribution[subject] = record.exercise_count
        
        # 计算平均每题用时
        average_time = total_time / max(1, total_exercises)
        
        # 计算准确率
        accuracy_rate = correct_exercises / max(1, total_exercises)
        
        # 知识点掌握情况
        knowledge_mastery = {}
        if student_model:
            knowledge_mastery = {
                k: v.mastery_level for k, v in student_model.knowledge_mastery.items()
            }
        
        # 学习趋势和每日活动
        daily_activity = {}
        learning_trend = {}
        
        # 获取历史数据
        history = self.get_learning_history(user.id, self.config.learning_trend_days)
        for date, data in history.items():
            daily_activity[date] = data.get("exercise_count", 0)
            learning_trend[date] = data.get("accuracy_rate", 0.0)
        
        # 整理推荐关注知识点
        recommended_focus = []
        if student_model:
            # 根据掌握度和重要性排序知识点
            sorted_kps = sorted(
                [(k, v.mastery_level) for k, v in student_model.knowledge_mastery.items()],
                key=lambda x: x[1]
            )
            recommended_focus = [kp for kp, _ in sorted_kps[:self.config.max_recommended_focus]]
        
        return LearningStats(
            total_exercises=total_exercises,
            correct_exercises=correct_exercises,
            accuracy_rate=accuracy_rate,
            total_time=total_time,
            average_time_per_exercise=average_time,
            subjects_distribution=subjects_distribution,
            knowledge_mastery=knowledge_mastery,
            daily_activity=daily_activity,
            learning_trend=learning_trend,
            strong_points=user.profile.strong_points,
            weak_points=user.profile.weak_points,
            recommended_focus=recommended_focus
        )
    
    def _update_learning_history(
        self, 
        user_id: str, 
        exercise: Optional[Exercise], 
        is_correct: bool, 
        time_spent: float
    ) -> None:
        """更新学习历史"""
        if user_id not in self._learning_history:
            self._learning_history[user_id] = {}
        
        # 获取今天的日期
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today not in self._learning_history[user_id]:
            self._learning_history[user_id][today] = {
                "exercise_count": 0,
                "correct_count": 0,
                "total_time": 0.0,
                "subjects": {},
                "knowledge_points": set()
            }
        
        # 更新今天的学习记录
        record = self._learning_history[user_id][today]
        record["exercise_count"] += 1
        if is_correct:
            record["correct_count"] += 1
        record["total_time"] += time_spent
        
        # 更新学科分布
        if exercise:
            subject = exercise.subject
            if subject not in record["subjects"]:
                record["subjects"][subject] = 0
            record["subjects"][subject] += 1
            
            # 添加知识点
            for kp in exercise.knowledge_points:
                record["knowledge_points"].add(kp)
        
        # 计算准确率
        record["accuracy_rate"] = record["correct_count"] / record["exercise_count"]
        
        # 将知识点集合转换为列表
        record["knowledge_points"] = list(record["knowledge_points"])
        
        # 清理过期历史
        self._clean_old_history(user_id)
        
        # 保存学习历史
        self._save_learning_history(user_id)
    
    def _load_learning_history(self, user_id: str) -> None:
        """加载学习历史"""
        # 实际项目中，这里会从数据库加载
        # 这里简单模拟一个空的历史记录
        self._learning_history[user_id] = {}
    
    def _save_learning_history(self, user_id: str) -> None:
        """保存学习历史"""
        # 实际项目中，这里会保存到数据库
        # 这里简单打印日志
        logger.info(f"保存用户 {user_id} 的学习历史: {len(self._learning_history[user_id])} 天")
    
    def _clean_old_history(self, user_id: str) -> None:
        """清理过期的历史记录"""
        if user_id not in self._learning_history:
            return
        
        # 计算截止日期
        cutoff_date = (datetime.now() - timedelta(days=self.config.learning_history_max_days)).strftime("%Y-%m-%d")
        
        # 清理旧数据
        self._learning_history[user_id] = {
            date: data for date, data in self._learning_history[user_id].items()
            if date >= cutoff_date
        }
    
    def _get_knowledge_point_info(self, kp_id: str) -> Dict[str, Any]:
        """获取知识点信息"""
        # 实际项目中，这里会从课标知识体系服务获取
        # 这里返回简单的模拟数据
        parts = kp_id.split("_")
        subject = parts[1] if len(parts) > 1 else "unknown"
        
        return {
            "id": kp_id,
            "name": f"知识点 {kp_id}",
            "subject": subject,
            "grade_level": 1,
            "description": f"{kp_id} 的描述",
            "prerequisites": [],
            "related_points": []
        }
    
    def _plan_learning_path(
        self, 
        target_kp: str, 
        mastered_kps: List[str]
    ) -> List[Dict[str, Any]]:
        """规划学习路径"""
        # 实际项目中，这里会调用课标知识体系服务的学习路径规划API
        # 这里返回简单的模拟数据
        return [{
            "id": target_kp,
            "name": f"知识点 {target_kp}",
            "mastered": target_kp in mastered_kps,
            "importance": 0.8,
            "difficulty": 0.6
        }]
    
    def _start_auto_sync(self) -> None:
        """启动自动同步任务"""
        logger.info(f"启动自动同步任务，间隔: {self.config.sync_interval}秒")
        
        # 实际项目中，这里会启动定时任务
        # 这里简单打印日志
        logger.info("自动同步任务已启动")


def create_user_learning_integration(
    user_service: UserService,
    recommendation_engine: RecommendationEngine,
    config: Dict = None
) -> UserLearningIntegration:
    """
    创建用户学习整合实例
    
    Args:
        user_service: 用户服务
        recommendation_engine: 推荐引擎
        config: 配置字典
        
    Returns:
        用户学习整合实例
    """
    # 创建配置
    integration_config = UserLearningConfig(**(config or {}))
    
    # 创建用户学习整合
    integration = UserLearningIntegration(
        user_service=user_service,
        recommendation_engine=recommendation_engine,
        config=integration_config
    )
    
    return integration 