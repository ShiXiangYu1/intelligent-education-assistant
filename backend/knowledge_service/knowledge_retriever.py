#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
知识检索模块

该模块实现了关键词-向量混合检索功能，用于从知识库中检索符合条件的内容。
核心功能包括:
1. 关键词检索: 基于BM25算法的精确关键词匹配
2. 向量检索: 基于语义相似度的内容匹配
3. 混合排序: 结合关键词和向量检索结果的综合排序
4. 年级过滤: 确保检索结果不超出指定年级范围
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from .vector_db import FAISSVectorDB, create_vector_db
from .keyword_index import ElasticsearchKeywordIndex, create_keyword_index


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeItem(BaseModel):
    """知识项数据模型"""
    id: str = Field(..., description="知识项唯一标识符")
    title: str = Field(..., description="知识项标题")
    content: str = Field(..., description="知识项内容")
    grade_level: int = Field(..., description="适用年级(1-12)")
    subject: str = Field(..., description="学科")
    keywords: List[str] = Field(default=[], description="关键词列表")
    vector: Optional[List[float]] = Field(default=None, description="向量表示")
    source: str = Field(default="", description="来源")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    updated_at: float = Field(default_factory=time.time, description="更新时间")


class SearchResult(BaseModel):
    """检索结果数据模型"""
    item: KnowledgeItem = Field(..., description="知识项")
    score: float = Field(..., description="匹配得分")
    keyword_score: float = Field(default=0.0, description="关键词匹配得分")
    vector_score: float = Field(default=0.0, description="向量相似度得分")


class KnowledgeRetriever:
    """知识检索器，实现关键词-向量混合检索"""
    
    def __init__(
        self,
        vector_db_client: Optional[FAISSVectorDB] = None,
        keyword_index: Optional[ElasticsearchKeywordIndex] = None,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        enable_grade_filter: bool = True,
        enable_mock_data: bool = False
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
        """
        self.vector_db_client = vector_db_client
        self.keyword_index = keyword_index
        self.enable_mock_data = enable_mock_data
        
        # 确保权重和为1
        total_weight = vector_weight + keyword_weight
        self.vector_weight = vector_weight / total_weight
        self.keyword_weight = keyword_weight / total_weight
        
        self.enable_grade_filter = enable_grade_filter
        logger.info(
            f"初始化知识检索器 - 向量权重: {self.vector_weight:.2f}, "
            f"关键词权重: {self.keyword_weight:.2f}, "
            f"年级过滤: {'启用' if enable_grade_filter else '禁用'}, "
            f"模拟数据: {'启用' if enable_mock_data else '禁用'}"
        )
    
    def retrieve(
        self,
        query: str,
        grade_level: Optional[int] = None,
        subject: Optional[str] = None,
        top_k: int = 10,
        min_score: float = 0.6
    ) -> List[SearchResult]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            grade_level: 年级限制(1-12)，None表示不限制
            subject: 学科限制，None表示不限制
            top_k: 返回的最大结果数量
            min_score: 最小匹配分数阈值
            
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
        
        # 4. 返回指定数量的结果
        top_results = sorted_results[:top_k]
        
        logger.info(f"检索完成 - 找到{len(top_results)}条结果")
        return top_results
    
    def _keyword_search(
        self,
        query: str,
        grade_level: Optional[int],
        subject: Optional[str],
        limit: int
    ) -> List[SearchResult]:
        """
        执行关键词检索
        
        Args:
            query: 查询文本
            grade_level: 年级限制
            subject: 学科限制
            limit: 返回结果数量限制
            
        Returns:
            关键词检索结果列表
        """
        if self.keyword_index is None:
            logger.warning("关键词索引未配置，跳过关键词检索")
            return []
        
        try:
            logger.debug(f"执行关键词检索: {query}")
            
            # 调用实际的关键词索引搜索
            es_results = self.keyword_index.search(
                query=query,
                grade_level=grade_level,
                subject=subject,
                top_k=limit
            )
            
            # 转换结果格式
            results = []
            for item_dict, score in es_results:
                # 确保字段类型正确
                if 'keywords' in item_dict and isinstance(item_dict['keywords'], str):
                    item_dict['keywords'] = item_dict['keywords'].split(',')
                
                # 创建KnowledgeItem对象
                item = KnowledgeItem(**item_dict)
                
                # 创建SearchResult对象
                search_result = SearchResult(
                    item=item,
                    score=score,
                    keyword_score=score,
                    vector_score=0.0
                )
                results.append(search_result)
            
            logger.debug(f"关键词检索完成，找到{len(results)}条结果")
            return results
        except Exception as e:
            logger.error(f"关键词检索出错: {str(e)}", exc_info=True)
            return []
    
    def _vector_search(
        self,
        query: str,
        grade_level: Optional[int],
        subject: Optional[str],
        limit: int
    ) -> List[SearchResult]:
        """
        执行向量检索
        
        Args:
            query: 查询文本
            grade_level: 年级限制
            subject: 学科限制
            limit: 返回结果数量限制
            
        Returns:
            向量检索结果列表
        """
        if self.vector_db_client is None:
            logger.warning("向量数据库未配置，跳过向量检索")
            return []
        
        try:
            logger.debug(f"执行向量检索: {query}")
            
            # 生成查询向量
            # 注意：实际项目中，这里需要调用文本嵌入模型生成向量
            # 这里假设已经有一个函数可以生成向量
            query_vector = self._get_embedding_vector(query)
            if query_vector is None:
                logger.warning("无法生成查询向量，跳过向量检索")
                return []
            
            # 调用向量数据库搜索
            vector_results = self.vector_db_client.search(
                query_vector=query_vector,
                top_k=limit
            )
            
            # 获取对应的知识项
            results = []
            for item_id, score in vector_results:
                # 获取完整的知识项信息
                item_data = self._get_knowledge_item_by_id(item_id)
                if item_data:
                    # 应用学科和年级过滤
                    if (subject is None or item_data.get('subject') == subject) and \
                       (grade_level is None or item_data.get('grade_level') == grade_level):
                        
                        # 创建KnowledgeItem对象
                        item = KnowledgeItem(**item_data)
                        
                        # 创建SearchResult对象
                        search_result = SearchResult(
                            item=item,
                            score=score,
                            keyword_score=0.0,
                            vector_score=score
                        )
                        results.append(search_result)
            
            logger.debug(f"向量检索完成，找到{len(results)}条结果")
            return results
        except Exception as e:
            logger.error(f"向量检索出错: {str(e)}", exc_info=True)
            return []
    
    def _get_embedding_vector(self, text: str) -> Optional[List[float]]:
        """
        获取文本的嵌入向量
        
        Args:
            text: 要嵌入的文本
            
        Returns:
            嵌入向量或None(如果生成失败)
        """
        # 实际项目中，这里应该调用文本嵌入模型
        # 这里返回一个随机向量作为示例
        try:
            # 假设向量维度是768(BERT base)
            dimension = 768
            if hasattr(self.vector_db_client, 'dimension'):
                dimension = self.vector_db_client.dimension
            
            # 生成随机向量并归一化
            vector = np.random.randn(dimension).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            
            return vector.tolist()
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {str(e)}", exc_info=True)
            return None
    
    def _get_knowledge_item_by_id(self, item_id: str) -> Optional[Dict]:
        """
        根据ID获取知识项完整信息
        
        Args:
            item_id: 知识项ID
            
        Returns:
            知识项字典或None(如果不存在)
        """
        # 实际项目中，这里应该从数据库或缓存中获取知识项
        # 这里我们直接从关键词索引获取（如果可用）
        try:
            if self.keyword_index and self.keyword_index.es:
                try:
                    result = self.keyword_index.es.get(
                        index=self.keyword_index.index_name,
                        id=item_id
                    )
                    if result and '_source' in result:
                        return result['_source']
                except Exception as e:
                    logger.debug(f"从关键词索引获取知识项失败: {str(e)}")
            
            # 如果从关键词索引获取失败，返回模拟数据
            if self.enable_mock_data:
                return {
                    "id": item_id,
                    "title": f"知识项 {item_id}",
                    "content": f"这是ID为{item_id}的知识项内容。",
                    "grade_level": 5,
                    "subject": "数学",
                    "keywords": ["示例", "模拟数据"],
                    "source": "模拟数据",
                    "created_at": time.time(),
                    "updated_at": time.time()
                }
            
            return None
        except Exception as e:
            logger.error(f"获取知识项失败: {str(e)}", exc_info=True)
            return None
    
    def _merge_results(
        self,
        keyword_results: List[SearchResult],
        vector_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        合并关键词检索和向量检索结果
        
        Args:
            keyword_results: 关键词检索结果
            vector_results: 向量检索结果
            
        Returns:
            合并后的结果列表
        """
        # 创建ID到结果的映射
        result_map = {}
        
        # 处理关键词检索结果
        for result in keyword_results:
            result_map[result.item.id] = result
        
        # 处理向量检索结果，合并相同ID的结果
        for result in vector_results:
            if result.item.id in result_map:
                # 已存在同ID的结果，需要合并
                existing_result = result_map[result.item.id]
                
                # 记录各自的得分
                existing_result.vector_score = result.vector_score
                
                # 计算加权总分
                combined_score = (
                    self.keyword_weight * existing_result.keyword_score +
                    self.vector_weight * result.vector_score
                )
                
                # 更新总分
                existing_result.score = combined_score
            else:
                # 新结果，直接添加到映射
                result_map[result.item.id] = result
        
        # 返回所有合并后的结果
        return list(result_map.values())
    
    def _apply_filters(
        self,
        item: KnowledgeItem,
        grade_level: Optional[int],
        subject: Optional[str]
    ) -> bool:
        """
        应用过滤条件
        
        Args:
            item: 知识项
            grade_level: 年级限制
            subject: 学科限制
            
        Returns:
            是否通过过滤
        """
        # 学科过滤
        if subject is not None and item.subject != subject:
            return False
        
        # 年级过滤(如果启用)
        if self.enable_grade_filter and grade_level is not None:
            if item.grade_level > grade_level:
                # 知识项年级高于目标年级，不符合要求
                return False
        
        return True
    
    # 以下是模拟数据方法，仅在没有实际数据源或调试时使用
    
    def _mock_keyword_search(
        self,
        query: str,
        grade_level: Optional[int],
        subject: Optional[str],
        limit: int
    ) -> List[SearchResult]:
        """模拟关键词检索结果"""
        logger.debug(f"使用模拟关键词检索: {query}")
        
        # 模拟关键词检索结果
        mock_results = []
        for i in range(5):
            item = KnowledgeItem(
                id=f"keyword_{i}",
                title=f"关键词检索结果 {i}",
                content=f"这是关键词检索到的第{i}个结果，与查询'{query}'相关",
                grade_level=grade_level or 5,
                subject=subject or "数学",
                keywords=query.split()
            )
            score = 0.9 - (i * 0.1)  # 模拟分数递减
            mock_results.append(
                SearchResult(
                    item=item, 
                    score=score,
                    keyword_score=score,
                    vector_score=0.0
                )
            )
            
        return mock_results
    
    def _mock_vector_search(
        self,
        query: str,
        grade_level: Optional[int],
        subject: Optional[str],
        limit: int
    ) -> List[SearchResult]:
        """模拟向量检索结果"""
        logger.debug(f"使用模拟向量检索: {query}")
        
        # 模拟向量检索结果
        mock_results = []
        for i in range(5):
            item = KnowledgeItem(
                id=f"vector_{i}",
                title=f"向量检索结果 {i}",
                content=f"这是向量检索到的第{i}个结果，与查询'{query}'语义相关",
                grade_level=grade_level or 5,
                subject=subject or "数学",
                keywords=[],
                vector=[0.1] * 10  # 模拟10维向量
            )
            score = 0.95 - (i * 0.1)  # 模拟分数递减
            mock_results.append(
                SearchResult(
                    item=item, 
                    score=score,
                    keyword_score=0.0,
                    vector_score=score
                )
            )
            
        return mock_results


def create_retriever(
    vector_db_config: Dict = None,
    keyword_index_config: Dict = None,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
    enable_grade_filter: bool = True,
    enable_mock_data: bool = False
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
    
    # 创建并返回检索器
    return KnowledgeRetriever(
        vector_db_client=vector_db_client,
        keyword_index=keyword_index,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        enable_grade_filter=enable_grade_filter,
        enable_mock_data=enable_mock_data
    ) 