#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于课标的内容过滤器模块

该模块提供根据课标知识体系对内容进行过滤和评估的功能，
确保检索和生成的内容符合特定年级和学科的课标要求。
"""

import os
import re
import json
import time
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union

from .knowledge_model import Subject, GradeLevel, KnowledgePoint
from .curriculum_service import CurriculumService, create_curriculum_service


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContentFilterConfig:
    """内容过滤器配置类"""
    
    def __init__(
        self,
        strictness_level: float = 0.7,
        enable_keyword_matching: bool = True,
        enable_concept_matching: bool = True,
        min_matching_keywords: int = 2,
        min_matching_ratio: float = 0.3,
        cache_size: int = 1000,
        curriculum_service_config: Dict = None
    ):
        """
        初始化内容过滤器配置
        
        Args:
            strictness_level: 过滤严格程度 (0-1)，数值越高要求匹配越严格
            enable_keyword_matching: 是否启用关键词匹配
            enable_concept_matching: 是否启用概念匹配
            min_matching_keywords: 最小匹配关键词数量
            min_matching_ratio: 最小匹配比例
            cache_size: 缓存大小
            curriculum_service_config: 课程体系服务配置
        """
        self.strictness_level = strictness_level
        self.enable_keyword_matching = enable_keyword_matching
        self.enable_concept_matching = enable_concept_matching
        self.min_matching_keywords = min_matching_keywords
        self.min_matching_ratio = min_matching_ratio
        self.cache_size = cache_size
        self.curriculum_service_config = curriculum_service_config or {}


class ContentEvaluationResult:
    """内容评估结果类"""
    
    def __init__(
        self,
        content_id: str,
        is_appropriate: bool,
        grade_appropriate: bool,
        subject_appropriate: bool,
        matched_keywords: List[str],
        matched_knowledge_points: List[str],
        confidence_score: float,
        grade_level: GradeLevel,
        subject: Subject,
        evaluation_time: float
    ):
        """
        初始化内容评估结果
        
        Args:
            content_id: 内容ID
            is_appropriate: 内容是否适合目标年级和学科
            grade_appropriate: 内容是否适合目标年级
            subject_appropriate: 内容是否适合目标学科
            matched_keywords: 匹配的关键词列表
            matched_knowledge_points: 匹配的知识点ID列表
            confidence_score: 置信度分数 (0-1)
            grade_level: 评估的目标年级
            subject: 评估的目标学科
            evaluation_time: 评估耗时 (秒)
        """
        self.content_id = content_id
        self.is_appropriate = is_appropriate
        self.grade_appropriate = grade_appropriate
        self.subject_appropriate = subject_appropriate
        self.matched_keywords = matched_keywords
        self.matched_knowledge_points = matched_knowledge_points
        self.confidence_score = confidence_score
        self.grade_level = grade_level
        self.subject = subject
        self.evaluation_time = evaluation_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "content_id": self.content_id,
            "is_appropriate": self.is_appropriate,
            "grade_appropriate": self.grade_appropriate,
            "subject_appropriate": self.subject_appropriate,
            "matched_keywords": self.matched_keywords,
            "matched_knowledge_points": self.matched_knowledge_points,
            "confidence_score": self.confidence_score,
            "grade_level": self.grade_level,
            "subject": self.subject,
            "evaluation_time": self.evaluation_time
        }


class ContentFilter:
    """基于课标的内容过滤器类"""
    
    def __init__(self, config: ContentFilterConfig):
        """
        初始化内容过滤器
        
        Args:
            config: 内容过滤器配置
        """
        self.config = config
        
        # 初始化课程体系服务
        self.curriculum_service = create_curriculum_service(config.curriculum_service_config)
        
        # 初始化缓存
        self.evaluation_cache = {}
        self.cache_keys = []
        
        logger.info("内容过滤器初始化完成")
    
    def evaluate_content(
        self,
        content: str,
        subject: Subject,
        grade_level: GradeLevel,
        content_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> ContentEvaluationResult:
        """
        评估内容是否符合特定年级和学科的课标要求
        
        Args:
            content: 待评估的内容文本
            subject: 目标学科
            grade_level: 目标年级
            content_id: 内容ID (可选)
            metadata: 内容元数据 (可选)
            
        Returns:
            内容评估结果
        """
        start_time = time.time()
        
        # 如果未提供内容ID，则生成一个临时ID
        if content_id is None:
            content_id = f"content_{int(start_time)}_{hash(content) % 10000}"
        
        # 查询缓存
        cache_key = f"{content_id}_{subject}_{grade_level}"
        if cache_key in self.evaluation_cache:
            logger.debug(f"从缓存中获取评估结果: {cache_key}")
            return self.evaluation_cache[cache_key]
        
        # 获取该学科和年级的知识点
        knowledge_points = self.curriculum_service.get_knowledge_points_by_subject_and_grade(
            subject, grade_level
        )
        
        if not knowledge_points:
            logger.warning(f"未找到学科 {subject} 年级 {grade_level} 的知识点")
            return ContentEvaluationResult(
                content_id=content_id,
                is_appropriate=False,
                grade_appropriate=False,
                subject_appropriate=False,
                matched_keywords=[],
                matched_knowledge_points=[],
                confidence_score=0.0,
                grade_level=grade_level,
                subject=subject,
                evaluation_time=time.time() - start_time
            )
        
        # 提取所有知识点的关键词
        all_keywords = set()
        for kp in knowledge_points:
            all_keywords.update([kp.name.lower()])
            all_keywords.update([kw.lower() for kw in kp.keywords])
        
        # 提取内容中匹配的关键词
        matched_keywords = []
        content_lower = content.lower()
        
        for keyword in all_keywords:
            if keyword and len(keyword) > 1 and keyword in content_lower:
                # 使用正则表达式确保匹配的是完整单词
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, content_lower):
                    matched_keywords.append(keyword)
        
        # 找到匹配关键词对应的知识点
        matched_kp_ids = set()
        for kp in knowledge_points:
            if kp.name.lower() in matched_keywords:
                matched_kp_ids.add(kp.id)
            else:
                for kw in kp.keywords:
                    if kw.lower() in matched_keywords:
                        matched_kp_ids.add(kp.id)
                        break
        
        # 根据匹配结果和配置计算置信度分数
        if len(matched_keywords) >= self.config.min_matching_keywords:
            # 计算匹配率
            match_ratio = len(matched_keywords) / max(len(all_keywords), 1)
            
            # 根据匹配关键词数量和匹配率计算置信度
            confidence_score = min(
                (len(matched_keywords) / max(self.config.min_matching_keywords, 1)) * 0.5 +
                (match_ratio / max(self.config.min_matching_ratio, 0.01)) * 0.5,
                1.0
            )
            
            # 根据严格程度确定是否通过
            is_appropriate = confidence_score >= self.config.strictness_level
            grade_appropriate = is_appropriate
            subject_appropriate = True
        else:
            confidence_score = 0.0
            is_appropriate = False
            grade_appropriate = False
            subject_appropriate = True
        
        # 创建评估结果
        result = ContentEvaluationResult(
            content_id=content_id,
            is_appropriate=is_appropriate,
            grade_appropriate=grade_appropriate,
            subject_appropriate=subject_appropriate,
            matched_keywords=matched_keywords,
            matched_knowledge_points=list(matched_kp_ids),
            confidence_score=confidence_score,
            grade_level=grade_level,
            subject=subject,
            evaluation_time=time.time() - start_time
        )
        
        # 更新缓存
        self._update_cache(cache_key, result)
        
        return result
    
    def filter_contents(
        self,
        contents: List[Dict[str, Any]],
        subject: Subject,
        grade_level: GradeLevel,
        min_confidence: float = None
    ) -> List[Dict[str, Any]]:
        """
        过滤内容列表，仅保留符合课标要求的内容
        
        Args:
            contents: 内容列表，每个内容为包含'id'和'text'键的字典
            subject: 目标学科
            grade_level: 目标年级
            min_confidence: 最小置信度 (可选，默认使用配置中的严格度)
            
        Returns:
            过滤后的内容列表
        """
        if min_confidence is None:
            min_confidence = self.config.strictness_level
        
        filtered_contents = []
        
        for content_item in contents:
            content_id = content_item.get('id', None)
            content_text = content_item.get('text', '')
            
            if not content_text:
                logger.warning(f"内容项缺少文本: {content_id}")
                continue
            
            # 评估内容
            evaluation = self.evaluate_content(
                content=content_text,
                subject=subject,
                grade_level=grade_level,
                content_id=content_id,
                metadata=content_item.get('metadata', {})
            )
            
            # 如果内容适合且置信度满足要求，则添加到结果中
            if evaluation.is_appropriate and evaluation.confidence_score >= min_confidence:
                # 添加评估信息到内容项
                content_item['evaluation'] = evaluation.to_dict()
                filtered_contents.append(content_item)
        
        logger.info(f"过滤内容: 总共{len(contents)}项，符合要求{len(filtered_contents)}项")
        return filtered_contents
    
    def classify_content_grade(
        self, 
        content: str,
        subject: Subject
    ) -> Tuple[GradeLevel, float]:
        """
        根据内容预测其最适合的年级
        
        Args:
            content: 内容文本
            subject: 学科
            
        Returns:
            (预测年级, 置信度)
        """
        best_grade = None
        best_confidence = 0.0
        
        # 获取所有年级
        all_grades = list(GradeLevel)
        
        # 分别评估内容对每个年级的适合度
        for grade in all_grades:
            evaluation = self.evaluate_content(
                content=content,
                subject=subject,
                grade_level=grade
            )
            
            # 保留置信度最高的年级
            if evaluation.confidence_score > best_confidence:
                best_confidence = evaluation.confidence_score
                best_grade = grade
        
        # 如果所有年级的置信度都很低，返回None
        if best_confidence < 0.3:
            return None, best_confidence
        
        return best_grade, best_confidence
    
    def extract_knowledge_points(
        self,
        content: str,
        subject: Subject,
        grade_level: GradeLevel
    ) -> List[Dict[str, Any]]:
        """
        从内容中提取相关的知识点
        
        Args:
            content: 内容文本
            subject: 学科
            grade_level: 年级
            
        Returns:
            提取的知识点列表
        """
        # 评估内容
        evaluation = self.evaluate_content(
            content=content,
            subject=subject,
            grade_level=grade_level
        )
        
        # 获取匹配的知识点详情
        knowledge_points = []
        for kp_id in evaluation.matched_knowledge_points:
            kp = self.curriculum_service.get_knowledge_point(kp_id)
            if kp:
                knowledge_points.append({
                    "id": kp.id,
                    "name": kp.name,
                    "description": kp.description,
                    "difficulty": kp.difficulty,
                    "importance": kp.importance
                })
        
        return knowledge_points
    
    def is_above_grade(
        self,
        content: str,
        subject: Subject,
        grade_level: GradeLevel
    ) -> Tuple[bool, float]:
        """
        判断内容是否超出指定年级的认知水平
        
        Args:
            content: 内容文本
            subject: 学科
            grade_level: 年级
            
        Returns:
            (是否超纲, 置信度)
        """
        # 预测内容的年级
        predicted_grade, confidence = self.classify_content_grade(content, subject)
        
        if predicted_grade is None:
            return False, 0.0
        
        # 将枚举转换为数字进行比较
        current_grade_num = GradeLevel.to_numeric(grade_level)
        predicted_grade_num = GradeLevel.to_numeric(predicted_grade)
        
        # 如果预测年级高于当前年级，则认为内容超纲
        is_above = predicted_grade_num > current_grade_num
        
        return is_above, confidence
    
    def suggest_content_modifications(
        self,
        content: str,
        subject: Subject,
        grade_level: GradeLevel
    ) -> Dict[str, Any]:
        """
        建议内容修改，使其更符合指定年级的课标
        
        Args:
            content: 内容文本
            subject: 学科
            grade_level: 年级
            
        Returns:
            内容修改建议
        """
        # 评估当前内容
        evaluation = self.evaluate_content(
            content=content,
            subject=subject,
            grade_level=grade_level
        )
        
        # 检查内容是否超纲
        is_above, above_confidence = self.is_above_grade(content, subject, grade_level)
        
        # 获取年级知识点
        grade_kps = self.curriculum_service.get_knowledge_points_by_subject_and_grade(
            subject, grade_level
        )
        
        # 准备建议
        suggestions = {
            "is_appropriate": evaluation.is_appropriate,
            "confidence_score": evaluation.confidence_score,
            "is_above_grade": is_above,
            "suggestions": []
        }
        
        # 如果内容适合，不需要修改
        if evaluation.is_appropriate and not is_above:
            suggestions["suggestions"].append({
                "type": "confirmation",
                "message": f"内容符合{grade_level}年级{subject}学科的课标要求，无需修改。"
            })
            return suggestions
        
        # 如果内容不适合或超纲，提供修改建议
        if not evaluation.is_appropriate:
            # 缺少关键知识点的情况
            suggestions["suggestions"].append({
                "type": "missing_knowledge",
                "message": f"内容缺少{grade_level}年级{subject}学科的关键知识点。",
                "recommended_knowledge_points": [
                    {
                        "id": kp.id,
                        "name": kp.name,
                        "importance": kp.importance
                    }
                    for kp in sorted(grade_kps, key=lambda k: k.importance, reverse=True)[:5]
                ]
            })
        
        if is_above:
            # 内容超纲的情况
            suggestions["suggestions"].append({
                "type": "above_grade",
                "message": f"内容包含{grade_level}年级学生可能难以理解的概念。",
                "simplification_tips": [
                    "简化术语和概念",
                    "减少复杂例子",
                    "调整表达方式使其更符合年级认知水平"
                ]
            })
        
        # 返回建议
        return suggestions
    
    def _update_cache(self, key: str, value: ContentEvaluationResult) -> None:
        """更新缓存"""
        # 如果缓存已满，移除最早的项
        if len(self.cache_keys) >= self.config.cache_size:
            oldest_key = self.cache_keys.pop(0)
            if oldest_key in self.evaluation_cache:
                del self.evaluation_cache[oldest_key]
        
        # 添加新项
        self.evaluation_cache[key] = value
        self.cache_keys.append(key)


def create_content_filter(config: Dict = None) -> ContentFilter:
    """
    创建内容过滤器实例
    
    Args:
        config: 配置字典
        
    Returns:
        内容过滤器实例
    """
    if config is None:
        config = {}
    
    filter_config = ContentFilterConfig(
        strictness_level=config.get('strictness_level', 0.7),
        enable_keyword_matching=config.get('enable_keyword_matching', True),
        enable_concept_matching=config.get('enable_concept_matching', True),
        min_matching_keywords=config.get('min_matching_keywords', 2),
        min_matching_ratio=config.get('min_matching_ratio', 0.3),
        cache_size=config.get('cache_size', 1000),
        curriculum_service_config=config.get('curriculum_service_config', {})
    )
    
    return ContentFilter(filter_config) 