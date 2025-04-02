#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
内容质量评估模块

该模块提供对教育内容质量的评估功能，支持多维度评估和质量改进建议。
"""

import os
import re
import time
import json
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union

import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QualityDimension:
    """质量维度枚举类"""
    ACCURACY = "accuracy"  # 准确性
    COMPLETENESS = "completeness"  # 完整性
    CLARITY = "clarity"  # 清晰度
    RELEVANCE = "relevance"  # 相关性
    STRUCTURE = "structure"  # 结构性
    ENGAGEMENT = "engagement"  # 吸引力
    AGE_APPROPRIATENESS = "age_appropriateness"  # 年龄适应性
    LANGUAGE = "language"  # 语言规范性


class ContentQualityConfig:
    """内容质量评估配置类"""
    
    def __init__(
        self,
        enabled_dimensions: List[str] = None,
        dimension_weights: Dict[str, float] = None,
        min_quality_score: float = 0.6,
        cache_size: int = 1000,
        use_external_api: bool = False,
        external_api_config: Dict[str, Any] = None
    ):
        """
        初始化内容质量评估配置
        
        Args:
            enabled_dimensions: 启用的质量维度列表
            dimension_weights: 各维度权重
            min_quality_score: 最低质量分数
            cache_size: 缓存大小
            use_external_api: 是否使用外部API进行评估
            external_api_config: 外部API配置
        """
        # 如果未指定启用的维度，则默认启用所有维度
        if enabled_dimensions is None:
            self.enabled_dimensions = [
                QualityDimension.ACCURACY,
                QualityDimension.COMPLETENESS,
                QualityDimension.CLARITY,
                QualityDimension.RELEVANCE,
                QualityDimension.STRUCTURE,
                QualityDimension.ENGAGEMENT,
                QualityDimension.AGE_APPROPRIATENESS,
                QualityDimension.LANGUAGE
            ]
        else:
            self.enabled_dimensions = enabled_dimensions
        
        # 如果未指定维度权重，则平均分配
        if dimension_weights is None:
            weight = 1.0 / len(self.enabled_dimensions)
            self.dimension_weights = {dim: weight for dim in self.enabled_dimensions}
        else:
            # 标准化权重，使总和为1
            total_weight = sum(dimension_weights.values())
            self.dimension_weights = {
                dim: weight / total_weight 
                for dim, weight in dimension_weights.items()
            }
        
        self.min_quality_score = min_quality_score
        self.cache_size = cache_size
        self.use_external_api = use_external_api
        self.external_api_config = external_api_config or {}


class QualityEvaluationResult:
    """质量评估结果类"""
    
    def __init__(
        self,
        content_id: str,
        overall_score: float,
        dimension_scores: Dict[str, float],
        improvement_suggestions: Dict[str, List[str]],
        evaluation_time: float
    ):
        """
        初始化质量评估结果
        
        Args:
            content_id: 内容ID
            overall_score: 总体质量分数(0-1)
            dimension_scores: 各维度的分数
            improvement_suggestions: 各维度的改进建议
            evaluation_time: 评估耗时(秒)
        """
        self.content_id = content_id
        self.overall_score = overall_score
        self.dimension_scores = dimension_scores
        self.improvement_suggestions = improvement_suggestions
        self.evaluation_time = evaluation_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "content_id": self.content_id,
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "improvement_suggestions": self.improvement_suggestions,
            "evaluation_time": self.evaluation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityEvaluationResult':
        """从字典创建评估结果对象"""
        return cls(
            content_id=data["content_id"],
            overall_score=data["overall_score"],
            dimension_scores=data["dimension_scores"],
            improvement_suggestions=data["improvement_suggestions"],
            evaluation_time=data["evaluation_time"]
        )


class ContentQualityEvaluator:
    """内容质量评估器类"""
    
    def __init__(self, config: ContentQualityConfig):
        """
        初始化内容质量评估器
        
        Args:
            config: 质量评估配置
        """
        self.config = config
        
        # 初始化缓存
        self.evaluation_cache = {}
        self.cache_keys = []
        
        # 初始化各维度的评估器字典
        self.dimension_evaluators = {
            QualityDimension.ACCURACY: self._evaluate_accuracy,
            QualityDimension.COMPLETENESS: self._evaluate_completeness,
            QualityDimension.CLARITY: self._evaluate_clarity,
            QualityDimension.RELEVANCE: self._evaluate_relevance,
            QualityDimension.STRUCTURE: self._evaluate_structure,
            QualityDimension.ENGAGEMENT: self._evaluate_engagement,
            QualityDimension.AGE_APPROPRIATENESS: self._evaluate_age_appropriateness,
            QualityDimension.LANGUAGE: self._evaluate_language
        }
        
        logger.info("内容质量评估器初始化完成")
    
    def evaluate_content(
        self,
        content: str,
        content_id: str = None,
        metadata: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> QualityEvaluationResult:
        """
        评估内容质量
        
        Args:
            content: 待评估的内容文本
            content_id: 内容ID(可选)
            metadata: 内容元数据(可选)
            context: 评估上下文信息(可选)
            
        Returns:
            质量评估结果
        """
        start_time = time.time()
        
        # 如果未提供内容ID，则生成一个临时ID
        if content_id is None:
            content_id = f"content_{int(start_time)}_{hash(content) % 10000}"
        
        # 查询缓存
        cache_key = f"{content_id}"
        if cache_key in self.evaluation_cache:
            logger.debug(f"从缓存中获取评估结果: {cache_key}")
            return self.evaluation_cache[cache_key]
        
        # 初始化评估结果
        dimension_scores = {}
        improvement_suggestions = {}
        
        # 评估各个维度
        for dimension in self.config.enabled_dimensions:
            if dimension in self.dimension_evaluators:
                # 调用对应维度的评估方法
                score, suggestions = self.dimension_evaluators[dimension](
                    content, metadata, context
                )
                dimension_scores[dimension] = score
                improvement_suggestions[dimension] = suggestions
        
        # 计算总体分数
        overall_score = 0.0
        for dimension, score in dimension_scores.items():
            weight = self.config.dimension_weights.get(dimension, 0)
            overall_score += score * weight
        
        # 创建评估结果
        result = QualityEvaluationResult(
            content_id=content_id,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            improvement_suggestions=improvement_suggestions,
            evaluation_time=time.time() - start_time
        )
        
        # 更新缓存
        self._update_cache(cache_key, result)
        
        return result
    
    def classify_content_quality(
        self,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        对内容质量进行分类
        
        Args:
            content: 内容文本
            metadata: 内容元数据(可选)
            
        Returns:
            质量分类: "excellent", "good", "adequate", "poor"
        """
        # 评估内容
        result = self.evaluate_content(content, metadata=metadata)
        
        # 根据总体分数分类
        if result.overall_score >= 0.85:
            return "excellent"
        elif result.overall_score >= 0.7:
            return "good"
        elif result.overall_score >= self.config.min_quality_score:
            return "adequate"
        else:
            return "poor"
    
    def get_improvement_plan(
        self,
        content: str,
        content_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        获取内容质量改进计划
        
        Args:
            content: 内容文本
            content_id: 内容ID(可选)
            metadata: 内容元数据(可选)
            
        Returns:
            改进计划字典，包含优先级排序的建议
        """
        # 评估内容
        result = self.evaluate_content(content, content_id, metadata)
        
        # 找出得分最低的几个维度
        sorted_dimensions = sorted(
            result.dimension_scores.items(), 
            key=lambda x: x[1]
        )
        
        # 准备改进计划
        improvement_plan = {
            "content_id": result.content_id,
            "overall_score": result.overall_score,
            "quality_classification": self.classify_content_quality(content, metadata),
            "prioritized_dimensions": [],
            "detailed_suggestions": {}
        }
        
        # 添加优先改进的维度
        for dimension, score in sorted_dimensions:
            if score < 0.7:  # 低于0.7分的维度需要优先改进
                improvement_plan["prioritized_dimensions"].append({
                    "dimension": dimension,
                    "score": score,
                    "suggestions": result.improvement_suggestions.get(dimension, [])
                })
                
                improvement_plan["detailed_suggestions"][dimension] = {
                    "score": score,
                    "suggestions": result.improvement_suggestions.get(dimension, []),
                    "priority": "high" if score < 0.5 else "medium"
                }
        
        return improvement_plan
    
    def _update_cache(self, key: str, value: QualityEvaluationResult) -> None:
        """更新缓存"""
        # 如果缓存已满，移除最早的项
        if len(self.cache_keys) >= self.config.cache_size:
            oldest_key = self.cache_keys.pop(0)
            if oldest_key in self.evaluation_cache:
                del self.evaluation_cache[oldest_key]
        
        # 添加新项
        self.evaluation_cache[key] = value
        self.cache_keys.append(key)
    
    # 以下是各维度的评估方法
    
    def _evaluate_accuracy(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Tuple[float, List[str]]:
        """
        评估内容的准确性
        
        检查内容中的事实陈述是否准确，是否包含错误信息。
        
        Args:
            content: 内容文本
            metadata: 内容元数据
            context: 评估上下文
            
        Returns:
            (准确性分数, 改进建议列表)
        """
        # TODO: 实现基于规则和模式匹配的准确性检查
        # 在实际实现中，这可能需要知识库查询或外部API调用
        
        # 简单实现：检查是否包含不确定性表达
        score = 0.8  # 默认较高的准确性分数
        suggestions = []
        
        # 检查不确定性表达
        uncertainty_patterns = [
            r'\b可能\b', r'\b也许\b', r'\b大概\b', r'\b猜测\b', r'\b不确定\b',
            r'\b似乎\b', r'\b好像\b', r'\b或许\b'
        ]
        
        for pattern in uncertainty_patterns:
            if re.search(pattern, content):
                score -= 0.05  # 每发现一个不确定表达，扣分
                suggestions.append("减少不确定性表达，使用更加确定和权威的表述")
                break  # 只添加一次这类建议
        
        # 检查是否包含相互矛盾的陈述
        if '但是' in content and '然而' in content:
            score -= 0.05
            suggestions.append("避免内容中出现相互矛盾的表述")
        
        # 返回分数和建议
        score = max(0.0, min(1.0, score))  # 确保分数在0-1之间
        return score, list(set(suggestions))  # 去重建议
    
    def _evaluate_completeness(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Tuple[float, List[str]]:
        """
        评估内容的完整性
        
        检查内容是否涵盖了主题的关键方面，是否有重要信息缺失。
        
        Args:
            content: 内容文本
            metadata: 内容元数据
            context: 评估上下文
            
        Returns:
            (完整性分数, 改进建议列表)
        """
        score = 0.7  # 默认初始分数
        suggestions = []
        
        # 内容长度检查
        length = len(content)
        if length < 100:
            score -= 0.2
            suggestions.append("内容过短，建议扩充更多相关信息")
        elif length < 200:
            score -= 0.1
            suggestions.append("内容略短，可以适当添加更多细节")
        
        # 检查是否包含引言和总结
        has_introduction = bool(re.search(r'^.{0,200}(介绍|引言|概述|简介)', content))
        has_conclusion = bool(re.search(r'.{0,200}(总结|结论|小结).*$', content))
        
        if not has_introduction:
            score -= 0.1
            suggestions.append("缺少引言或概述部分，建议添加以帮助读者理解上下文")
        
        if not has_conclusion:
            score -= 0.1
            suggestions.append("缺少总结部分，建议添加以强化关键点")
        
        # 检查关键要素是否存在（示例化）
        expected_elements = context.get('expected_elements', []) if context else []
        for element in expected_elements:
            if element.lower() not in content.lower():
                score -= 0.1
                suggestions.append(f"缺少关键要素：{element}，建议补充相关内容")
        
        # 确保分数在0-1之间
        score = max(0.0, min(1.0, score))
        return score, list(set(suggestions))
    
    def _evaluate_clarity(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Tuple[float, List[str]]:
        """
        评估内容的清晰度
        
        检查内容表达是否清晰，是否易于理解。
        
        Args:
            content: 内容文本
            metadata: 内容元数据
            context: 评估上下文
            
        Returns:
            (清晰度分数, 改进建议列表)
        """
        score = 0.8  # 默认较高的清晰度分数
        suggestions = []
        
        # 检查句子长度
        sentences = re.split(r'[。！？]', content)
        long_sentences = [s for s in sentences if len(s) > 50]
        if len(long_sentences) > len(sentences) * 0.3:
            score -= 0.1
            suggestions.append("句子过长，建议分解成更短、更易理解的句子")
        
        # 检查复杂词汇
        complex_words_count = 0
        # 这里可以添加专业词汇或复杂词汇的列表
        complex_words = ["抽象", "繁杂", "理论", "复杂", "深奥"]
        for word in complex_words:
            if word in content:
                complex_words_count += 1
        
        if complex_words_count > 5:
            score -= 0.1
            suggestions.append("使用了过多复杂词汇，建议用更简单的表达替代")
        
        # 检查是否使用了清晰的标题和小标题
        if not re.search(r'#+\s+\w+', content) and len(content) > 500:
            score -= 0.1
            suggestions.append("缺少清晰的标题结构，建议添加标题和小标题以提高可读性")
        
        # 检查解释性内容
        if "例如" not in content and "比如" not in content and len(content) > 300:
            score -= 0.05
            suggestions.append("缺少示例或解释性内容，建议添加具体例子以增强理解")
        
        # 确保分数在0-1之间
        score = max(0.0, min(1.0, score))
        return score, list(set(suggestions))
    
    def _evaluate_relevance(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Tuple[float, List[str]]:
        """
        评估内容的相关性
        
        检查内容是否与主题相关，是否包含不相关的信息。
        
        Args:
            content: 内容文本
            metadata: 内容元数据
            context: 评估上下文
            
        Returns:
            (相关性分数, 改进建议列表)
        """
        score = 0.75  # 默认相关性分数
        suggestions = []
        
        # 从上下文或元数据中获取主题关键词
        topic_keywords = []
        if context and 'topic_keywords' in context:
            topic_keywords = context['topic_keywords']
        elif metadata and 'keywords' in metadata:
            topic_keywords = metadata['keywords']
        
        # 检查主题关键词出现情况
        if topic_keywords:
            matched_keywords = 0
            for keyword in topic_keywords:
                if keyword.lower() in content.lower():
                    matched_keywords += 1
            
            keyword_match_ratio = matched_keywords / len(topic_keywords)
            if keyword_match_ratio < 0.5:
                score -= 0.2
                suggestions.append("内容与主题关键词匹配度低，建议增加与主题直接相关的内容")
        
        # 检查是否有离题段落
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1:
            for i, para in enumerate(paragraphs):
                if len(para) > 50:  # 只检查较长的段落
                    # 简单启发式：检查段落是否包含任何主题关键词
                    is_relevant = False
                    for keyword in topic_keywords:
                        if keyword.lower() in para.lower():
                            is_relevant = True
                            break
                    
                    if not is_relevant:
                        score -= 0.1
                        suggestions.append(f"第{i+1}段落可能离题，建议修改或删除")
                        break  # 只报告一次
        
        # 确保分数在0-1之间
        score = max(0.0, min(1.0, score))
        return score, list(set(suggestions))
    
    def _evaluate_structure(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Tuple[float, List[str]]:
        """
        评估内容的结构性
        
        检查内容结构是否合理，是否有逻辑层次。
        
        Args:
            content: 内容文本
            metadata: 内容元数据
            context: 评估上下文
            
        Returns:
            (结构性分数, 改进建议列表)
        """
        score = 0.7  # 默认结构性分数
        suggestions = []
        
        # 检查段落结构
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if len(paragraphs) < 3 and len(content) > 500:
            score -= 0.1
            suggestions.append("段落划分不足，建议适当增加段落以提高可读性")
        
        # 检查是否有明确的部分划分
        has_sections = False
        section_patterns = [
            r'#+\s+\w+',  # Markdown标题
            r'\d+\.\s+\w+',  # 数字编号
            r'[一二三四五六七八九十]+、'  # 中文编号
        ]
        
        for pattern in section_patterns:
            if re.search(pattern, content):
                has_sections = True
                break
        
        if not has_sections and len(content) > 500:
            score -= 0.1
            suggestions.append("缺少清晰的章节结构，建议添加标题或分节")
        
        # 检查逻辑顺序词
        logic_markers = ['首先', '其次', '然后', '最后', '因此', '所以', '总结']
        logic_marker_count = 0
        for marker in logic_markers:
            if marker in content:
                logic_marker_count += 1
        
        if logic_marker_count < 2 and len(content) > 300:
            score -= 0.1
            suggestions.append("缺少逻辑顺序标记，建议添加过渡词以增强内容连贯性")
        
        # 检查是否有明确的开头和结尾
        if not re.search(r'^.{0,200}(介绍|引言|概述)', content):
            score -= 0.05
            suggestions.append("缺少明确的开头部分，建议添加引言或概述")
        
        if not re.search(r'.{0,200}(总结|结论|总之).*$', content):
            score -= 0.05
            suggestions.append("缺少明确的结尾部分，建议添加总结或结论")
        
        # 确保分数在0-1之间
        score = max(0.0, min(1.0, score))
        return score, list(set(suggestions))
    
    def _evaluate_engagement(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Tuple[float, List[str]]:
        """
        评估内容的吸引力
        
        检查内容是否有趣，是否能吸引读者。
        
        Args:
            content: 内容文本
            metadata: 内容元数据
            context: 评估上下文
            
        Returns:
            (吸引力分数, 改进建议列表)
        """
        score = 0.6  # 默认吸引力分数
        suggestions = []
        
        # 检查是否使用了问题来吸引读者
        question_count = len(re.findall(r'[？\?]', content))
        if question_count == 0:
            score -= 0.1
            suggestions.append("缺少引人思考的问题，建议添加问题以增强互动性")
        
        # 检查是否使用了生动的例子
        example_patterns = [r'例如', r'比如', r'举例', r'案例']
        has_examples = False
        for pattern in example_patterns:
            if pattern in content:
                has_examples = True
                break
        
        if not has_examples:
            score -= 0.1
            suggestions.append("缺少生动的例子，建议添加具体示例以增强内容吸引力")
        
        # 检查语言多样性
        # 简单方法：计算不同词的比例
        words = re.findall(r'\w+', content)
        unique_words = set(words)
        word_diversity = len(unique_words) / max(1, len(words))
        
        if word_diversity < 0.5:
            score -= 0.1
            suggestions.append("词汇多样性不足，建议使用更丰富的表达方式")
        
        # 检查是否有情感元素
        emotion_words = ['有趣', '精彩', '惊人', '启发', '激动', '感动', '震撼']
        emotion_count = 0
        for word in emotion_words:
            if word in content:
                emotion_count += 1
        
        if emotion_count < 2:
            score -= 0.05
            suggestions.append("情感元素不足，建议适当添加情感表达以增强共鸣")
        
        # 检查是否有互动元素
        interaction_patterns = [r'思考', r'想想', r'尝试', r'试试', r'练习']
        has_interaction = False
        for pattern in interaction_patterns:
            if pattern in content:
                has_interaction = True
                break
        
        if not has_interaction:
            score -= 0.05
            suggestions.append("缺少互动元素，建议添加读者参与环节")
        
        # 确保分数在0-1之间
        score = max(0.0, min(1.0, score))
        return score, list(set(suggestions))
    
    def _evaluate_age_appropriateness(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Tuple[float, List[str]]:
        """
        评估内容的年龄适应性
        
        检查内容是否适合目标年龄段。
        
        Args:
            content: 内容文本
            metadata: 内容元数据
            context: 评估上下文
            
        Returns:
            (年龄适应性分数, 改进建议列表)
        """
        score = 0.8  # 默认年龄适应性分数
        suggestions = []
        
        # 从上下文或元数据中获取目标年级
        target_grade = None
        if context and 'grade_level' in context:
            target_grade = context['grade_level']
        elif metadata and 'grade_level' in metadata:
            target_grade = metadata['grade_level']
        
        if target_grade is None:
            # 无法评估年龄适应性
            return score, ["无法确定目标年级，无法评估年龄适应性"]
        
        # 根据年级检查词汇复杂度
        # 这里使用简化的实现，实际应使用更复杂的词汇分级系统
        complex_word_patterns = []
        
        if target_grade <= 3:  # 低年级
            complex_word_patterns = [
                r'抽象', r'理论', r'概念', r'原理', r'机制',
                r'系统性', r'逻辑性', r'分析', r'综合', r'评估'
            ]
        elif target_grade <= 6:  # 中年级
            complex_word_patterns = [
                r'哲学', r'理论', r'系统性', r'综合性', r'辩证',
                r'命题', r'假设', r'推导', r'论证', r'批判'
            ]
        
        if complex_word_patterns:
            for pattern in complex_word_patterns:
                if re.search(pattern, content):
                    score -= 0.05
                    suggestions.append(f"包含可能超出{target_grade}年级理解能力的概念，建议简化或提供更多解释")
                    break  # 只添加一次这类建议
        
        # 检查句子长度是否适合年级
        max_sentence_length = {
            1: 10, 2: 15, 3: 20, 4: 25, 5: 30, 6: 35,
            7: 40, 8: 45, 9: 50, 10: 55, 11: 60, 12: 65
        }
        
        sentences = re.split(r'[。！？]', content)
        long_sentences = [
            s for s in sentences 
            if len(s) > max_sentence_length.get(target_grade, 40)
        ]
        
        if len(long_sentences) > len(sentences) * 0.2:
            score -= 0.1
            suggestions.append(f"句子长度超过{target_grade}年级学生的理解能力，建议缩短句子或分解复杂句")
        
        # 检查内容是否包含适合年龄的例子和表述
        if target_grade <= 6:
            # 对于低年级，检查是否有具体、生活化的例子
            concrete_patterns = [r'例如', r'比如', r'生活中']
            has_concrete_examples = False
            
            for pattern in concrete_patterns:
                if pattern in content:
                    has_concrete_examples = True
                    break
            
            if not has_concrete_examples:
                score -= 0.1
                suggestions.append(f"缺少适合{target_grade}年级学生的具体例子，建议添加生活化、直观的示例")
        
        # 确保分数在0-1之间
        score = max(0.0, min(1.0, score))
        return score, list(set(suggestions))
    
    def _evaluate_language(
        self,
        content: str,
        metadata: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Tuple[float, List[str]]:
        """
        评估内容的语言规范性
        
        检查内容的语法、拼写和标点是否规范。
        
        Args:
            content: 内容文本
            metadata: 内容元数据
            context: 评估上下文
            
        Returns:
            (语言规范性分数, 改进建议列表)
        """
        score = 0.9  # 默认较高的语言规范性分数
        suggestions = []
        
        # 检查标点符号使用
        punctuation_errors = 0
        
        # 检查中英文标点混用
        mixed_punctuation_patterns = [
            r'[\u4e00-\u9fa5],', r'[\u4e00-\u9fa5]\.',  # 中文后接英文逗号或句号
            r'[a-zA-Z]，', r'[a-zA-Z]。'  # 英文后接中文逗号或句号
        ]
        
        for pattern in mixed_punctuation_patterns:
            if re.search(pattern, content):
                punctuation_errors += 1
        
        if punctuation_errors > 0:
            score -= 0.05
            suggestions.append("存在中英文标点混用问题，建议统一使用中文或英文标点")
        
        # 检查重复标点
        if re.search(r'[。，！？；：]{2,}', content):
            score -= 0.05
            suggestions.append("存在重复标点符号，建议规范使用标点")
        
        # 检查错别字（简单示例，实际应使用更完善的检查方法）
        typo_patterns = {
            r'\b的的\b': '的', r'\b地地\b': '地', r'\b得得\b': '得',
            r'\b是是\b': '是', r'\b了了\b': '了', r'\b和和\b': '和'
        }
        
        typo_count = 0
        for pattern, correction in typo_patterns.items():
            matches = re.findall(pattern, content)
            typo_count += len(matches)
        
        if typo_count > 0:
            score -= 0.1
            suggestions.append("存在潜在的错别字或重复用词，建议仔细检查")
        
        # 检查长难句
        sentences = re.split(r'[。！？]', content)
        complex_sentences = [s for s in sentences if len(s) > 50 and '，' in s]
        if len(complex_sentences) > len(sentences) * 0.3:
            score -= 0.05
            suggestions.append("存在过多长难句，建议适当分解以提高可读性")
        
        # 确保分数在0-1之间
        score = max(0.0, min(1.0, score))
        return score, list(set(suggestions))


def create_content_quality_evaluator(config: Dict = None) -> ContentQualityEvaluator:
    """
    创建内容质量评估器实例
    
    Args:
        config: 配置字典
        
    Returns:
        内容质量评估器实例
    """
    if config is None:
        config = {}
    
    # 从配置中提取参数
    enabled_dimensions = config.get('enabled_dimensions', None)
    dimension_weights = config.get('dimension_weights', None)
    min_quality_score = config.get('min_quality_score', 0.6)
    cache_size = config.get('cache_size', 1000)
    use_external_api = config.get('use_external_api', False)
    external_api_config = config.get('external_api_config', None)
    
    # 创建配置对象
    evaluator_config = ContentQualityConfig(
        enabled_dimensions=enabled_dimensions,
        dimension_weights=dimension_weights,
        min_quality_score=min_quality_score,
        cache_size=cache_size,
        use_external_api=use_external_api,
        external_api_config=external_api_config
    )
    
    # 创建并返回评估器
    return ContentQualityEvaluator(evaluator_config) 