#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
关键词索引模块

该模块实现了与Elasticsearch的集成，用于实现基于关键词的知识内容检索。
支持添加、删除、搜索等基本操作，以及高级的文本分析功能。
"""

import os
import time
import logging
import json
from typing import Dict, List, Optional, Tuple, Union, Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pydantic import BaseModel, Field

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KeywordIndexConfig(BaseModel):
    """关键词索引配置"""
    hosts: List[str] = Field(["http://localhost:9200"], description="Elasticsearch服务器地址")
    index_name: str = Field("knowledge_items", description="索引名称")
    username: Optional[str] = Field(None, description="ES用户名")
    password: Optional[str] = Field(None, description="ES密码")
    analyzer: str = Field("standard", description="分析器类型")
    timeout: int = Field(30, description="连接超时时间(秒)")
    max_retries: int = Field(3, description="最大重试次数")


class ElasticsearchKeywordIndex:
    """Elasticsearch关键词索引接口实现"""
    
    def __init__(self, config: KeywordIndexConfig):
        """
        初始化Elasticsearch关键词索引
        
        Args:
            config: 索引配置
        """
        self.config = config
        self.es = None
        self.index_name = config.index_name
        
        # 连接Elasticsearch
        self._connect()
        
        # 确保索引存在
        self._ensure_index()
        
        logger.info(f"初始化Elasticsearch关键词索引 - 索引: {self.index_name}")
        
    def _connect(self):
        """连接到Elasticsearch服务器"""
        try:
            # 构建连接参数
            es_args = {
                'hosts': self.config.hosts,
                'timeout': self.config.timeout,
                'retry_on_timeout': True,
                'max_retries': self.config.max_retries
            }
            
            # 添加认证信息(如果提供)
            if self.config.username and self.config.password:
                es_args['http_auth'] = (self.config.username, self.config.password)
            
            # 创建连接
            self.es = Elasticsearch(**es_args)
            
            # 检查连接
            if not self.es.ping():
                raise ConnectionError("无法连接到Elasticsearch服务器")
            
            logger.info(f"成功连接到Elasticsearch: {self.config.hosts}")
        except Exception as e:
            logger.error(f"连接Elasticsearch失败: {str(e)}", exc_info=True)
            self.es = None
    
    def _ensure_index(self):
        """确保索引存在，如不存在则创建"""
        if self.es is None:
            logger.error("Elasticsearch未连接，无法创建索引")
            return False
        
        try:
            # 检查索引是否存在
            if self.es.indices.exists(index=self.index_name):
                logger.info(f"索引'{self.index_name}'已存在")
                return True
            
            # 定义索引映射
            mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "analyzer": self.config.analyzer,
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "content": {
                            "type": "text",
                            "analyzer": self.config.analyzer
                        },
                        "grade_level": {"type": "integer"},
                        "subject": {"type": "keyword"},
                        "keywords": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "created_at": {"type": "date", "format": "epoch_second"},
                        "updated_at": {"type": "date", "format": "epoch_second"}
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "text_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop", "snowball"]
                            }
                        }
                    }
                }
            }
            
            # 创建索引
            self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"索引'{self.index_name}'创建成功")
            return True
        except Exception as e:
            logger.error(f"创建索引失败: {str(e)}", exc_info=True)
            return False
    
    def add(self, item: Dict[str, Any]) -> bool:
        """
        添加项目到索引
        
        Args:
            item: 要添加的知识项字典
            
        Returns:
            是否添加成功
        """
        if self.es is None:
            logger.error("Elasticsearch未连接，无法添加项目")
            return False
        
        # 确保必需字段存在
        required_fields = ["id", "title", "content"]
        for field in required_fields:
            if field not in item:
                logger.error(f"项目缺少必需字段: {field}")
                return False
        
        try:
            # 确保时间字段为epoch格式
            if "created_at" not in item:
                item["created_at"] = time.time()
            if "updated_at" not in item:
                item["updated_at"] = time.time()
            
            # 添加到索引
            self.es.index(
                index=self.index_name,
                id=item["id"],
                body=item,
                refresh=True  # 立即刷新索引以便搜索
            )
            
            logger.info(f"项目'{item['id']}'添加成功")
            return True
        except Exception as e:
            logger.error(f"添加项目失败: {str(e)}", exc_info=True)
            return False
    
    def bulk_add(self, items: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        批量添加项目到索引
        
        Args:
            items: 要添加的知识项列表
            
        Returns:
            (成功数量, 失败数量)
        """
        if self.es is None:
            logger.error("Elasticsearch未连接，无法批量添加项目")
            return (0, len(items))
        
        if not items:
            return (0, 0)
        
        try:
            # 准备批量操作
            actions = []
            for item in items:
                # 确保必需字段存在
                required_fields = ["id", "title", "content"]
                valid = True
                for field in required_fields:
                    if field not in item:
                        logger.warning(f"项目缺少必需字段: {field}，跳过")
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # 确保时间字段为epoch格式
                if "created_at" not in item:
                    item["created_at"] = time.time()
                if "updated_at" not in item:
                    item["updated_at"] = time.time()
                
                action = {
                    "_index": self.index_name,
                    "_id": item["id"],
                    "_source": item
                }
                actions.append(action)
            
            if not actions:
                logger.warning("没有有效的项目可添加")
                return (0, len(items))
            
            # 执行批量操作
            success, errors = bulk(
                self.es,
                actions,
                refresh=True,  # 立即刷新索引以便搜索
                stats_only=True
            )
            
            logger.info(f"批量添加完成 - 成功: {success}, 失败: {errors}")
            return (success, errors)
        except Exception as e:
            logger.error(f"批量添加失败: {str(e)}", exc_info=True)
            return (0, len(items))
    
    def search(
        self,
        query: str,
        grade_level: Optional[int] = None,
        subject: Optional[str] = None,
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        搜索符合条件的知识项
        
        Args:
            query: 搜索查询
            grade_level: 年级限制
            subject: 学科限制
            top_k: 返回的最大结果数量
            
        Returns:
            (知识项, 得分)列表，按得分降序排列
        """
        if self.es is None:
            logger.error("Elasticsearch未连接，无法搜索")
            return []
        
        try:
            # 构建搜索查询
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title^2", "content", "keywords^3"],
                                    "type": "best_fields",
                                    "operator": "or",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ],
                        "filter": []
                    }
                },
                "size": top_k,
                "_source": True
            }
            
            # 添加过滤条件
            if grade_level is not None:
                search_body["query"]["bool"]["filter"].append(
                    {"term": {"grade_level": grade_level}}
                )
            
            if subject is not None:
                search_body["query"]["bool"]["filter"].append(
                    {"term": {"subject": subject}}
                )
            
            # 执行搜索
            response = self.es.search(
                index=self.index_name,
                body=search_body
            )
            
            # 处理结果
            results = []
            for hit in response["hits"]["hits"]:
                item = hit["_source"]
                score = hit["_score"]
                # 归一化分数到0-1范围
                normalized_score = min(score / 10.0, 1.0)
                results.append((item, normalized_score))
            
            logger.info(f"搜索完成 - 查询: '{query}', 找到: {len(results)}个结果")
            return results
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}", exc_info=True)
            return []
    
    def delete(self, id: str) -> bool:
        """
        删除项目
        
        Args:
            id: 要删除的项目ID
            
        Returns:
            是否删除成功
        """
        if self.es is None:
            logger.error("Elasticsearch未连接，无法删除项目")
            return False
        
        try:
            # 检查项目是否存在
            if not self.es.exists(index=self.index_name, id=id):
                logger.warning(f"项目'{id}'不存在，无法删除")
                return False
            
            # 删除项目
            self.es.delete(
                index=self.index_name,
                id=id,
                refresh=True  # 立即刷新索引
            )
            
            logger.info(f"项目'{id}'删除成功")
            return True
        except Exception as e:
            logger.error(f"删除项目失败: {str(e)}", exc_info=True)
            return False
    
    def clear(self) -> bool:
        """
        清空索引
        
        Returns:
            是否清空成功
        """
        if self.es is None:
            logger.error("Elasticsearch未连接，无法清空索引")
            return False
        
        try:
            # 删除并重新创建索引
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
            
            # 重新创建索引
            self._ensure_index()
            
            logger.info(f"索引'{self.index_name}'已清空")
            return True
        except Exception as e:
            logger.error(f"清空索引失败: {str(e)}", exc_info=True)
            return False
    
    def stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息
        
        Returns:
            包含统计信息的字典
        """
        if self.es is None:
            return {"status": "未连接"}
        
        try:
            # 获取索引统计信息
            count_resp = self.es.count(index=self.index_name)
            stats_resp = self.es.indices.stats(index=self.index_name)
            
            return {
                "status": "已连接",
                "total_documents": count_resp["count"],
                "index_size_bytes": stats_resp["indices"][self.index_name]["total"]["store"]["size_in_bytes"],
                "index_name": self.index_name,
                "hosts": self.config.hosts
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}", exc_info=True)
            return {"status": "错误", "error": str(e)}


def create_keyword_index(config: Dict = None) -> ElasticsearchKeywordIndex:
    """
    创建关键词索引实例
    
    Args:
        config: 配置字典
        
    Returns:
        关键词索引实例
    """
    if config is None:
        config = {}
    
    index_config = KeywordIndexConfig(**config)
    return ElasticsearchKeywordIndex(index_config) 