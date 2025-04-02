#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
知识服务模块初始化文件

该模块提供了知识检索和内容生成功能，是智能教育助手系统的核心组件之一。
主要功能包括：
1. 知识检索：支持向量检索和关键词检索的混合检索策略
2. 内容生成：基于RAG的教育内容生成
3. 内容质量评估：多维度评估内容质量并提供改进建议
4. 课标过滤：确保内容符合课标要求
"""

from .knowledge_retriever import (
    KnowledgeRetriever, 
    KnowledgeItem, 
    SearchResult, 
    create_retriever
)

from .content_generator import (
    ContentGenerator,
    GenerationRequest,
    GeneratedContent,
    ContentSource,
    create_content_generator
)

from .content_quality_evaluator import (
    ContentQualityEvaluator,
    ContentQualityConfig,
    QualityEvaluationResult,
    QualityDimension,
    create_content_quality_evaluator
)

from .api import app, start_server

__all__ = [
    # 知识检索相关
    'KnowledgeRetriever',
    'KnowledgeItem',
    'SearchResult',
    'create_retriever',
    
    # 内容生成相关
    'ContentGenerator',
    'GenerationRequest',
    'GeneratedContent',
    'ContentSource',
    'create_content_generator',
    
    # 内容质量评估相关
    'ContentQualityEvaluator',
    'ContentQualityConfig',
    'QualityEvaluationResult',
    'QualityDimension',
    'create_content_quality_evaluator',
    
    # API服务相关
    'app',
    'start_server'
] 