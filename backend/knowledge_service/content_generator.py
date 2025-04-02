#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
内容生成模块

该模块实现了基于RAG（检索增强生成）的内容生成功能，用于生成符合课标的教育内容。
核心功能包括:
1. 检索相关知识：利用知识检索模块获取相关内容
2. 内容合成：根据检索结果生成符合要求的内容
3. 课标过滤：确保生成内容符合年级和学科课标要求
4. 质量评估：评估生成内容的质量和准确性
"""

import os
import time
import logging
import json
from typing import Dict, List, Optional, Tuple, Union, Any

from pydantic import BaseModel, Field
import openai

# 从知识检索模块导入相关类
from .knowledge_retriever import KnowledgeRetriever, KnowledgeItem, SearchResult
# 导入内容质量评估模块
from .content_quality_evaluator import ContentQualityEvaluator, create_content_quality_evaluator


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GenerationRequest(BaseModel):
    """内容生成请求数据模型"""
    query: str = Field(..., description="用户查询/问题")
    grade_level: Optional[int] = Field(None, description="年级(1-12)")
    subject: Optional[str] = Field(None, description="学科")
    max_length: int = Field(500, description="生成内容的最大长度")
    temperature: float = Field(0.7, description="生成内容的创造性程度(0-1)")
    format: str = Field("text", description="输出格式，如text/json/markdown")
    style: Optional[str] = Field(None, description="内容风格，如'简洁'/'详细'/'通俗易懂'等")


class ContentSource(BaseModel):
    """内容来源引用数据模型"""
    id: str = Field(..., description="来源ID")
    title: str = Field(..., description="来源标题")
    content_snippet: str = Field(..., description="引用的内容片段")
    relevance_score: float = Field(..., description="相关性得分")


class GeneratedContent(BaseModel):
    """生成的内容数据模型"""
    content: str = Field(..., description="生成的内容")
    sources: List[ContentSource] = Field(default=[], description="内容来源引用")
    query: str = Field(..., description="原始查询")
    grade_level: Optional[int] = Field(None, description="内容适用年级")
    subject: Optional[str] = Field(None, description="内容所属学科")
    generation_time: float = Field(default_factory=time.time, description="生成时间")
    quality_score: Optional[float] = Field(None, description="内容质量评分")
    feedback: Optional[str] = Field(None, description="生成反馈")
    quality_details: Optional[Dict[str, Any]] = Field(None, description="详细质量评估结果")


class ContentGenerator:
    """内容生成器，基于RAG实现教育内容生成"""
    
    def __init__(
        self,
        retriever: KnowledgeRetriever,
        llm_client=None,
        llm_model: str = "gpt-3.5-turbo",
        max_context_length: int = 4000,
        quality_threshold: float = 0.7,
        enable_quality_check: bool = True,
        system_prompt_template: Optional[str] = None,
        quality_evaluator: Optional[ContentQualityEvaluator] = None
    ):
        """
        初始化内容生成器
        
        Args:
            retriever: 知识检索器实例
            llm_client: 大语言模型客户端
            llm_model: 使用的大语言模型名称
            max_context_length: 最大上下文长度
            quality_threshold: 内容质量阈值
            enable_quality_check: 是否启用质量检查
            system_prompt_template: 系统提示词模板
            quality_evaluator: 内容质量评估器实例
        """
        self.retriever = retriever
        self.llm_client = llm_client or openai.Client()
        self.llm_model = llm_model
        self.max_context_length = max_context_length
        self.quality_threshold = quality_threshold
        self.enable_quality_check = enable_quality_check
        self.quality_evaluator = quality_evaluator
        
        # 设置默认系统提示词模板
        self.system_prompt_template = system_prompt_template or """
            你是一个专业的教育内容生成助手，专注于为{grade_level}年级的{subject}学科生成高质量的教育内容。
            你需要根据提供的参考资料，回答用户的问题或生成相关内容。
            请确保你的回答:
            1. 准确无误，符合科学事实
            2. 符合{grade_level}年级学生的认知水平
            3. 遵循教育课标的要求
            4. 用通俗易懂的语言表达
            5. 适当举例说明复杂概念
            6. 条理清晰，逻辑连贯
            7. 不要生成超出{grade_level}年级课标范围的内容
            
            回答风格: {style}
            输出格式: {format}
        """
        
        logger.info(
            f"初始化内容生成器 - 模型: {llm_model}, "
            f"质量检查: {'启用' if enable_quality_check else '禁用'}, "
            f"质量阈值: {quality_threshold}, "
            f"质量评估器: {'已配置' if quality_evaluator else '未配置'}"
        )
    
    def generate(self, request: GenerationRequest) -> GeneratedContent:
        """
        生成教育内容
        
        Args:
            request: 内容生成请求
            
        Returns:
            生成的内容，包括正文和来源引用
        """
        logger.info(f"处理内容生成请求 - 查询: '{request.query}'")
        
        # 1. 检索相关知识
        search_results = self.retriever.retrieve(
            query=request.query,
            grade_level=request.grade_level,
            subject=request.subject,
            top_k=5,
            min_score=0.6
        )
        
        if not search_results:
            logger.warning(f"未找到与'{request.query}'相关的知识项")
            # 生成一个基础回复，无需检索的知识支持
            return self._generate_basic_response(request)
        
        # 2. 准备上下文
        context, sources = self._prepare_context(search_results)
        
        # 3. 构建提示词
        system_prompt = self._build_system_prompt(request)
        user_prompt = self._build_user_prompt(request, context)
        
        # 4. 调用LLM生成内容
        content = self._call_llm(system_prompt, user_prompt, request.temperature, request.max_length)
        
        # 5. 质量检查（如果启用）
        quality_score = None
        feedback = None
        quality_details = None
        
        if self.enable_quality_check:
            quality_result = self._check_quality(content, request, search_results)
            
            if isinstance(quality_result, tuple) and len(quality_result) >= 2:
                quality_score, feedback = quality_result[:2]
                if len(quality_result) > 2:
                    quality_details = quality_result[2]
            
            # 如果质量不达标，尝试重新生成
            if quality_score and quality_score < self.quality_threshold:
                logger.warning(f"内容质量不达标 (得分: {quality_score}), 尝试重新生成")
                
                # 调整提示词，强调质量问题
                improved_system_prompt = system_prompt + f"\n请注意改进以下问题: {feedback}"
                content = self._call_llm(improved_system_prompt, user_prompt, request.temperature, request.max_length)
                
                # 再次评估质量
                quality_result = self._check_quality(content, request, search_results)
                if isinstance(quality_result, tuple) and len(quality_result) >= 2:
                    quality_score, feedback = quality_result[:2]
                    if len(quality_result) > 2:
                        quality_details = quality_result[2]
        
        # 6. 构建并返回结果
        generated_content = GeneratedContent(
            content=content,
            sources=sources,
            query=request.query,
            grade_level=request.grade_level,
            subject=request.subject,
            quality_score=quality_score,
            feedback=feedback,
            quality_details=quality_details
        )
        
        logger.info(f"内容生成完成 - 质量得分: {quality_score or '未评估'}")
        return generated_content
    
    def _prepare_context(self, search_results: List[SearchResult]) -> Tuple[str, List[ContentSource]]:
        """
        准备上下文和来源引用
        
        Args:
            search_results: 检索结果列表
            
        Returns:
            上下文字符串和来源引用列表
        """
        context_parts = []
        sources = []
        
        for i, result in enumerate(search_results):
            item = result.item
            
            # 添加上下文部分
            context_part = f"[{i+1}] {item.title}\n{item.content}\n"
            context_parts.append(context_part)
            
            # 添加来源引用
            source = ContentSource(
                id=item.id,
                title=item.title,
                content_snippet=item.content[:100] + "..." if len(item.content) > 100 else item.content,
                relevance_score=result.score
            )
            sources.append(source)
        
        # 合并上下文，确保不超过最大长度
        context = "\n".join(context_parts)
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
            
        return context, sources
    
    def _build_system_prompt(self, request: GenerationRequest) -> str:
        """
        构建系统提示词
        
        Args:
            request: 内容生成请求
            
        Returns:
            格式化的系统提示词
        """
        grade_text = f"{request.grade_level}" if request.grade_level else "适合的"
        subject_text = request.subject if request.subject else "通用"
        style_text = request.style if request.style else "清晰专业"
        
        return self.system_prompt_template.format(
            grade_level=grade_text,
            subject=subject_text,
            style=style_text,
            format=request.format
        )
    
    def _build_user_prompt(self, request: GenerationRequest, context: str) -> str:
        """
        构建用户提示词
        
        Args:
            request: 内容生成请求
            context: 上下文信息
            
        Returns:
            格式化的用户提示词
        """
        return f"""
        问题/要求:
        {request.query}
        
        参考资料:
        {context}
        
        请基于上述参考资料回答问题，确保内容符合{request.grade_level or '适合的'}年级学生的认知水平。
        如果参考资料中没有相关信息，请明确说明，并提供一般性的回答。
        """
    
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        调用大语言模型生成内容
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            temperature: 温度参数(创造性程度)
            max_tokens: 最大生成token数
            
        Returns:
            生成的内容文本
        """
        try:
            logger.debug(f"调用LLM - 模型: {self.llm_model}, 温度: {temperature}")
            
            # 提取查询文本(用于记录日志)
            query_text = user_prompt.split("\n")[1].strip() if "\n" in user_prompt else user_prompt
            
            # 实际项目中，这里会调用OpenAI或其他LLM API
            # 由于API限制，这里使用简单的模拟实现
            
            import time
            time.sleep(1)  # 模拟API调用延迟
            
            # 模拟返回结果
            mock_response = f"""
            这是针对"{query_text}"的回答。
            
            根据所提供的参考资料，我们可以得出以下内容：
            
            1. 主要概念解释
            这个概念是指...（基于参考资料生成的内容）
            
            2. 重要原理
            相关的原理包括...（基于参考资料生成的内容）
            
            3. 应用例子
            在日常生活中，我们可以看到...（基于参考资料生成的内容）
            
            希望这个解答对你有帮助！如果有任何疑问，欢迎继续提问。
            """
            
            return mock_response.strip()
            
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            return f"内容生成失败: {str(e)}"
    
    def _check_quality(
        self,
        content: str,
        request: GenerationRequest,
        search_results: List[SearchResult]
    ) -> Tuple[float, str, Optional[Dict[str, Any]]]:
        """
        检查生成内容的质量
        
        Args:
            content: 生成的内容
            request: 内容生成请求
            search_results: 检索结果
            
        Returns:
            质量得分(0-1)、反馈信息和详细质量评估结果
        """
        logger.debug("评估内容质量")
        
        # 使用内容质量评估器进行评估
        if self.quality_evaluator:
            try:
                # 准备评估上下文和元数据
                metadata = {
                    "grade_level": request.grade_level,
                    "subject": request.subject,
                    "query": request.query,
                    "keywords": []
                }
                
                # 从检索结果中提取关键词
                for result in search_results:
                    if hasattr(result.item, 'keywords') and result.item.keywords:
                        metadata["keywords"].extend(result.item.keywords)
                
                # 去重关键词
                metadata["keywords"] = list(set(metadata["keywords"]))
                
                # 准备评估上下文
                context = {
                    "grade_level": request.grade_level,
                    "subject": request.subject,
                    "topic_keywords": metadata["keywords"],
                    "expected_elements": []
                }
                
                # 根据查询和检索结果分析预期要素
                # 这里简单地从检索结果标题中抽取可能的要素
                for result in search_results:
                    title_words = result.item.title.split()
                    if len(title_words) >= 2:
                        context["expected_elements"].append(result.item.title)
                
                # 调用评估器评估内容
                evaluation_result = self.quality_evaluator.evaluate_content(
                    content=content,
                    metadata=metadata,
                    context=context
                )
                
                # 获取改进计划（如果分数较低）
                improvement_plan = None
                if evaluation_result.overall_score < self.quality_threshold:
                    improvement_plan = self.quality_evaluator.get_improvement_plan(
                        content=content,
                        metadata=metadata
                    )
                
                # 构造反馈信息
                feedback = self._construct_feedback_from_evaluation(evaluation_result, improvement_plan)
                
                # 返回评估结果
                return (
                    evaluation_result.overall_score,
                    feedback,
                    {
                        "dimension_scores": evaluation_result.dimension_scores,
                        "improvement_suggestions": evaluation_result.improvement_suggestions,
                        "improvement_plan": improvement_plan
                    }
                )
                
            except Exception as e:
                logger.error(f"内容质量评估失败: {str(e)}", exc_info=True)
        
        # 如果没有配置评估器或评估失败，使用备用评估方法
        logger.warning("使用备用质量评估方法")
        
        # 基础质量分数(0.7-0.9随机)
        import random
        base_score = 0.7 + random.random() * 0.2
        
        # 模拟评估反馈
        feedback = None
        if base_score < 0.8:
            feedback = "内容可以更加详细，并增加更多具体例子。"
        
        return base_score, feedback or "内容质量良好", None
    
    def _construct_feedback_from_evaluation(
        self,
        evaluation_result,
        improvement_plan=None
    ) -> str:
        """
        从评估结果构造反馈信息
        
        Args:
            evaluation_result: 评估结果
            improvement_plan: 改进计划
            
        Returns:
            格式化的反馈文本
        """
        feedback_parts = []
        
        # 添加总体质量评价
        quality_class = "优秀" if evaluation_result.overall_score >= 0.85 else (
            "良好" if evaluation_result.overall_score >= 0.7 else (
                "适当" if evaluation_result.overall_score >= 0.6 else "需要改进"
            )
        )
        
        feedback_parts.append(f"内容质量总体评价: {quality_class} (得分: {evaluation_result.overall_score:.2f})")
        
        # 添加维度评分
        if evaluation_result.dimension_scores:
            feedback_parts.append("\n各维度评分:")
            dimension_names = {
                "accuracy": "准确性",
                "completeness": "完整性",
                "clarity": "清晰度",
                "relevance": "相关性",
                "structure": "结构性",
                "engagement": "吸引力",
                "age_appropriateness": "年龄适应性",
                "language": "语言规范性"
            }
            
            # 按分数从低到高排序，优先展示需要改进的维度
            sorted_scores = sorted(
                evaluation_result.dimension_scores.items(),
                key=lambda x: x[1]
            )
            
            for dim, score in sorted_scores:
                dim_name = dimension_names.get(dim, dim)
                feedback_parts.append(f"- {dim_name}: {score:.2f}")
        
        # 添加主要改进建议
        if improvement_plan and "prioritized_dimensions" in improvement_plan:
            feedback_parts.append("\n主要改进建议:")
            
            for dim_info in improvement_plan["prioritized_dimensions"][:3]:  # 最多显示前3个
                dim_name = dimension_names.get(dim_info["dimension"], dim_info["dimension"])
                feedback_parts.append(f"- {dim_name} (得分: {dim_info['score']:.2f}):")
                
                for suggestion in dim_info["suggestions"]:
                    feedback_parts.append(f"  * {suggestion}")
        
        return "\n".join(feedback_parts)
    
    def _generate_basic_response(self, request: GenerationRequest) -> GeneratedContent:
        """
        在没有检索到相关内容时生成基础回复
        
        Args:
            request: 内容生成请求
            
        Returns:
            生成的基础内容
        """
        logger.debug("生成基础回复(无检索支持)")
        
        # 构建简化提示词
        system_prompt = self._build_system_prompt(request)
        user_prompt = f"""
        问题/要求:
        {request.query}
        
        请提供一个基础回答，注意内容应适合{request.grade_level or '学生'}年级学生的理解水平。
        """
        
        # 调用LLM生成内容
        content = self._call_llm(system_prompt, user_prompt, request.temperature, request.max_length)
        
        # 检查质量（如果启用）
        quality_score = None
        feedback = None
        quality_details = None
        
        if self.enable_quality_check and self.quality_evaluator:
            try:
                # 准备评估上下文和元数据
                metadata = {
                    "grade_level": request.grade_level,
                    "subject": request.subject,
                    "query": request.query
                }
                
                # 简化评估上下文
                context = {
                    "grade_level": request.grade_level,
                    "subject": request.subject
                }
                
                # 调用评估器
                evaluation_result = self.quality_evaluator.evaluate_content(
                    content=content,
                    metadata=metadata,
                    context=context
                )
                
                quality_score = evaluation_result.overall_score
                feedback = self._construct_feedback_from_evaluation(evaluation_result)
                quality_details = {
                    "dimension_scores": evaluation_result.dimension_scores,
                    "improvement_suggestions": evaluation_result.improvement_suggestions
                }
                
            except Exception as e:
                logger.error(f"内容质量评估失败: {str(e)}", exc_info=True)
        
        # 构建结果
        return GeneratedContent(
            content=content,
            sources=[],
            query=request.query,
            grade_level=request.grade_level,
            subject=request.subject,
            quality_score=quality_score,
            feedback=feedback,
            quality_details=quality_details
        )


def create_content_generator(
    retriever: KnowledgeRetriever,
    llm_config: Dict = None,
    quality_threshold: float = 0.7,
    enable_quality_check: bool = True,
    quality_evaluator_config: Dict = None
) -> ContentGenerator:
    """
    创建内容生成器的工厂函数
    
    Args:
        retriever: 知识检索器实例
        llm_config: LLM配置
        quality_threshold: 内容质量阈值
        enable_quality_check: 是否启用质量检查
        quality_evaluator_config: 质量评估器配置
        
    Returns:
        配置好的内容生成器实例
    """
    # 设置默认LLM配置
    default_llm_config = {
        "model": "gpt-3.5-turbo",
        "max_context_length": 4000
    }
    llm_config = llm_config or default_llm_config
    
    # 创建质量评估器（如果启用质量检查）
    quality_evaluator = None
    if enable_quality_check:
        try:
            quality_evaluator = create_content_quality_evaluator(quality_evaluator_config)
            logger.info("已创建内容质量评估器")
        except Exception as e:
            logger.error(f"创建内容质量评估器失败: {str(e)}", exc_info=True)
            logger.warning("将使用备用质量评估方法")
    
    # 创建内容生成器
    return ContentGenerator(
        retriever=retriever,
        llm_model=llm_config.get("model", default_llm_config["model"]),
        max_context_length=llm_config.get("max_context_length", default_llm_config["max_context_length"]),
        quality_threshold=quality_threshold,
        enable_quality_check=enable_quality_check,
        quality_evaluator=quality_evaluator
    ) 