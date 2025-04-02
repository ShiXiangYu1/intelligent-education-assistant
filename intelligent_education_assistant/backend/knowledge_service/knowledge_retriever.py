import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from .vector_db import FAISSVectorDB, create_vector_db
from .keyword_index import ElasticsearchKeywordIndex, create_keyword_index

# 导入内容过滤器
try:
    from backend.curriculum_system.content_filter import ContentFilter, create_content_filter
    HAS_CONTENT_FILTER = True
except ImportError:
    HAS_CONTENT_FILTER = False
    logging.warning("未找到内容过滤器模块，课标过滤功能将被禁用")

class KnowledgeRetriever:
    """知识检索器，实现关键词-向量混合检索"""
    
    def __init__(
        self,
        vector_db_client: Optional[FAISSVectorDB] = None,
        keyword_index: Optional[ElasticsearchKeywordIndex] = None,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        enable_grade_filter: bool = True,
        enable_mock_data: bool = False,
        content_filter: Optional['ContentFilter'] = None
    ):
        """
        初始化知识检索器
        
        Args:
            vector_db_client: 向量数据库客户端
            keyword_index: 关键词索引客户端
            vector_weight: 向量检索结果权重 (0-1)
            keyword_weight: 关键词检索结果权重 (0-1)
            enable_grade_filter: 是否启用年级过滤
            enable_mock_data: 当实际检索结果为空时是否返回模拟数据
            content_filter: 基于课标的内容过滤器
        """
        self.vector_db_client = vector_db_client
        self.keyword_index = keyword_index
        self.enable_mock_data = enable_mock_data
        self.content_filter = content_filter
        
        # 确保权重和为1
        total_weight = vector_weight + keyword_weight
        self.vector_weight = vector_weight / total_weight
        self.keyword_weight = keyword_weight / total_weight
        
        self.enable_grade_filter = enable_grade_filter
        logger.info(
            f"初始化知识检索器 - 向量权重: {self.vector_weight:.2f}, "
            f"关键词权重: {self.keyword_weight:.2f}, "
            f"年级过滤: {'启用' if enable_grade_filter else '禁用'}, "
            f"模拟数据: {'启用' if enable_mock_data else '禁用'}, "
            f"课标过滤: {'启用' if content_filter is not None else '禁用'}"
        )
    
    def retrieve(
        self,
        query: str,
        grade_level: Optional[int] = None,
        subject: Optional[str] = None,
        top_k: int = 10,
        min_score: float = 0.6,
        apply_curriculum_filter: bool = True
    ) -> List[SearchResult]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            grade_level: 年级限制(1-12)，None表示不限制
            subject: 学科限制，None表示不限制
            top_k: 返回的最大结果数量
            min_score: 最小匹配分数阈值
            apply_curriculum_filter: 是否应用课标过滤
            
        Returns:
            符合条件的检索结果列表
        """
        logger.info(f"执行检索 - 查询: '{query}', 年级: {grade_level}, 学科: {subject}")
        
        # 1. 分别执行关键词检索和向量检索
        keyword_results = self._keyword_search(query, grade_level, subject, top_k * 2)
        vector_results = self._vector_search(query, grade_level, subject, top_k * 2)
        
        # 检查是否有实际检索结果，如果没有且启用了模拟数据，则使用模拟数据
        if not keyword_results and not vector_results and self.enable_mock_data:
            logger.warning("没有找到实际检索结果，使用模拟数据")
            keyword_results = self._mock_keyword_search(query, grade_level, subject, top_k)
            vector_results = self._mock_vector_search(query, grade_level, subject, top_k)
        
        # 2. 合并检索结果
        merged_results = self._merge_results(keyword_results, vector_results)
        
        # 3. 过滤和排序结果
        filtered_results = [
            result for result in merged_results 
            if result.score >= min_score and self._apply_filters(result.item, grade_level, subject)
        ]
        sorted_results = sorted(filtered_results, key=lambda x: x.score, reverse=True)
        
        # 4. 应用课标内容过滤（如果启用）
        if apply_curriculum_filter and self.content_filter is not None and HAS_CONTENT_FILTER:
            sorted_results = self._apply_curriculum_filter(sorted_results, subject, grade_level)
        
        # 5. 返回指定数量的结果
        top_results = sorted_results[:top_k]
        
        logger.info(f"检索完成 - 找到{len(top_results)}条结果")
        return top_results

    def _apply_curriculum_filter(
        self,
        results: List[SearchResult],
        subject: Optional[str],
        grade_level: Optional[int]
    ) -> List[SearchResult]:
        """
        应用课标内容过滤器
        
        Args:
            results: 检索结果列表
            subject: 学科
            grade_level: 年级
            
        Returns:
            经课标过滤后的结果列表
        """
        if not results or self.content_filter is None or not HAS_CONTENT_FILTER:
            return results
        
        try:
            # 转换学科和年级为课标过滤器使用的枚举类型
            # 注意：为简化实现，这里假设subject和GradeLevel.from_numeric的映射关系
            # 实际项目中需要进行更精确的转换
            from backend.curriculum_system.knowledge_model import Subject, GradeLevel
            
            try:
                curriculum_subject = getattr(Subject, subject.upper()) if subject else None
            except (AttributeError, KeyError):
                curriculum_subject = None
                logger.warning(f"无法将学科 '{subject}' 映射到课标系统中的学科")
            
            curriculum_grade = None
            if grade_level is not None:
                curriculum_grade = GradeLevel.from_numeric(grade_level)
                if curriculum_grade is None:
                    logger.warning(f"无法将年级 {grade_level} 映射到课标系统中的年级")
            
            # 如果无法映射学科或年级，返回原结果
            if curriculum_subject is None or curriculum_grade is None:
                return results
            
            filtered_results = []
            logger.info(f"应用课标过滤 - 学科: {curriculum_subject}, 年级: {curriculum_grade}")
            
            for result in results:
                # 为每个结果应用课标过滤
                evaluation = self.content_filter.evaluate_content(
                    content=result.item.content,
                    subject=curriculum_subject,
                    grade_level=curriculum_grade
                )
                
                # 保留符合课标要求的结果
                if evaluation.is_appropriate:
                    # 调整分数，考虑课标适合度
                    result.score = result.score * 0.7 + evaluation.confidence_score * 0.3
                    # 添加课标评估信息到knowledge_item的metadata
                    if result.item.metadata is None:
                        result.item.metadata = {}
                    result.item.metadata['curriculum_evaluation'] = evaluation.to_dict()
                    filtered_results.append(result)
                else:
                    logger.debug(f"知识项 {result.item.id} 不符合课标要求，已过滤")
            
            logger.info(f"课标过滤结果 - 过滤前: {len(results)}, 过滤后: {len(filtered_results)}")
            return filtered_results
        except Exception as e:
            logger.error(f"应用课标过滤时出错: {str(e)}", exc_info=True)
            return results

def create_retriever(
    vector_db_config: Dict = None,
    keyword_index_config: Dict = None,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
    enable_grade_filter: bool = True,
    enable_mock_data: bool = False,
    content_filter_config: Dict = None
) -> KnowledgeRetriever:
    """
    创建知识检索器实例
    
    Args:
        vector_db_config: 向量数据库配置
        keyword_index_config: 关键词索引配置
        vector_weight: 向量检索权重
        keyword_weight: 关键词检索权重
        enable_grade_filter: 是否启用年级过滤
        enable_mock_data: 当实际检索结果为空时是否返回模拟数据
        content_filter_config: 内容过滤器配置
        
    Returns:
        知识检索器实例
    """
    logger.info("创建知识检索器实例")
    
    # 创建向量数据库客户端(如果提供了配置)
    vector_db_client = None
    if vector_db_config is not None:
        try:
            vector_db_client = create_vector_db(vector_db_config)
            logger.info("成功创建向量数据库客户端")
        except Exception as e:
            logger.error(f"创建向量数据库客户端失败: {str(e)}", exc_info=True)
    
    # 创建关键词索引客户端(如果提供了配置)
    keyword_index = None
    if keyword_index_config is not None:
        try:
            keyword_index = create_keyword_index(keyword_index_config)
            logger.info("成功创建关键词索引客户端")
        except Exception as e:
            logger.error(f"创建关键词索引客户端失败: {str(e)}", exc_info=True)
    
    # 创建内容过滤器(如果提供了配置且模块可用)
    content_filter = None
    if content_filter_config is not None and HAS_CONTENT_FILTER:
        try:
            content_filter = create_content_filter(content_filter_config)
            logger.info("成功创建内容过滤器")
        except Exception as e:
            logger.error(f"创建内容过滤器失败: {str(e)}", exc_info=True)
    
    # 创建并返回检索器
    return KnowledgeRetriever(
        vector_db_client=vector_db_client,
        keyword_index=keyword_index,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        enable_grade_filter=enable_grade_filter,
        enable_mock_data=enable_mock_data,
        content_filter=content_filter
    ) 