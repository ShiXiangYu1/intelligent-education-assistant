#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
知识服务API模块

该模块实现了知识服务的API接口，包括知识检索和内容生成功能。
使用FastAPI框架提供RESTful API服务。
"""

import os
import time
import logging
import json
from typing import Dict, List, Optional, Any, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .knowledge_retriever import (
    KnowledgeRetriever, KnowledgeItem, SearchResult, create_retriever
)
from .content_generator import (
    ContentGenerator, GenerationRequest, GeneratedContent,
    ContentSource, create_content_generator
)


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 创建FastAPI应用
app = FastAPI(
    title="智能教育助手 - 知识服务API",
    description="提供知识检索和内容生成功能的API接口",
    version="0.1.0",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应当限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API请求和响应模型
class SearchRequest(BaseModel):
    """知识检索请求模型"""
    query: str = Field(..., description="检索查询")
    grade_level: Optional[int] = Field(None, description="年级(1-12)")
    subject: Optional[str] = Field(None, description="学科")
    top_k: int = Field(10, description="返回结果数量")
    min_score: float = Field(0.6, description="最小匹配分数")


class SearchResponse(BaseModel):
    """知识检索响应模型"""
    items: List[Dict[str, Any]] = Field(..., description="检索结果列表")
    total: int = Field(..., description="结果总数")
    query: str = Field(..., description="原始查询")
    search_time: float = Field(..., description="检索耗时(秒)")


class ContentRequest(BaseModel):
    """内容生成请求模型"""
    query: str = Field(..., description="用户查询/问题")
    grade_level: Optional[int] = Field(None, description="年级(1-12)")
    subject: Optional[str] = Field(None, description="学科")
    max_length: int = Field(500, description="生成内容的最大长度")
    temperature: float = Field(0.7, description="生成内容的创造性程度(0-1)")
    format: str = Field("text", description="输出格式，可选text/json/markdown")
    style: Optional[str] = Field(None, description="内容风格，如'简洁'/'详细'等")


class ContentResponse(BaseModel):
    """内容生成响应模型"""
    content: str = Field(..., description="生成的内容")
    sources: List[Dict[str, Any]] = Field(..., description="内容来源引用")
    quality_score: Optional[float] = Field(None, description="内容质量得分")
    generation_time: float = Field(..., description="生成耗时(秒)")


class AddKnowledgeItemRequest(BaseModel):
    """添加知识项请求模型"""
    title: str = Field(..., description="知识项标题")
    content: str = Field(..., description="知识项内容")
    grade_level: int = Field(..., description="适用年级(1-12)")
    subject: str = Field(..., description="学科")
    keywords: List[str] = Field(default=[], description="关键词列表")
    source: str = Field(default="", description="来源")


class AddKnowledgeItemResponse(BaseModel):
    """添加知识项响应模型"""
    id: str = Field(..., description="知识项ID")
    status: str = Field(..., description="添加状态")
    vector_db_status: bool = Field(..., description="向量数据库添加状态")
    keyword_index_status: bool = Field(..., description="关键词索引添加状态")


# 依赖项：获取知识检索器实例
def get_retriever() -> KnowledgeRetriever:
    """
    获取知识检索器实例
    
    这里使用单例模式，确保应用中只创建一个检索器实例
    """
    # 实际项目中，这里应该从配置中读取参数
    if not hasattr(get_retriever, "instance"):
        logger.info("创建知识检索器实例")
        
        # 从环境变量或配置文件加载配置
        try:
            # 向量数据库配置
            vector_db_config = {
                "dimension": int(os.getenv("VECTOR_DB_DIMENSION", "768")),
                "index_type": os.getenv("VECTOR_DB_INDEX_TYPE", "Flat"),
                "use_gpu": os.getenv("VECTOR_DB_USE_GPU", "false").lower() == "true",
                "db_path": os.getenv("VECTOR_DB_PATH", "./data/vector_db")
            }
            
            # 关键词索引配置
            keyword_index_config = {
                "hosts": os.getenv("ES_HOSTS", "http://localhost:9200").split(","),
                "index_name": os.getenv("ES_INDEX_NAME", "knowledge_items"),
                "username": os.getenv("ES_USERNAME", None),
                "password": os.getenv("ES_PASSWORD", None)
            }
            
            # 检索器配置
            vector_weight = float(os.getenv("RETRIEVER_VECTOR_WEIGHT", "0.6"))
            keyword_weight = float(os.getenv("RETRIEVER_KEYWORD_WEIGHT", "0.4"))
            enable_grade_filter = os.getenv("RETRIEVER_ENABLE_GRADE_FILTER", "true").lower() == "true"
            enable_mock_data = os.getenv("RETRIEVER_ENABLE_MOCK_DATA", "true").lower() == "true"
            
            # 创建检索器
            get_retriever.instance = create_retriever(
                vector_db_config=vector_db_config,
                keyword_index_config=keyword_index_config,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                enable_grade_filter=enable_grade_filter,
                enable_mock_data=enable_mock_data
            )
            
            logger.info("检索器创建成功，配置如下:")
            logger.info(f"向量权重: {vector_weight}, 关键词权重: {keyword_weight}")
            logger.info(f"年级过滤: {'启用' if enable_grade_filter else '禁用'}")
            logger.info(f"模拟数据: {'启用' if enable_mock_data else '禁用'}")
        except Exception as e:
            logger.error(f"创建检索器失败: {str(e)}", exc_info=True)
            # 创建一个基础的检索器
            get_retriever.instance = create_retriever(enable_mock_data=True)
            logger.info("已创建基础检索器（使用模拟数据）")
    
    return get_retriever.instance


# 依赖项：获取内容生成器实例
def get_generator(retriever: KnowledgeRetriever = Depends(get_retriever)) -> ContentGenerator:
    """
    获取内容生成器实例
    
    这里使用单例模式，确保应用中只创建一个生成器实例
    """
    # 实际项目中，这里应该从配置中读取参数
    if not hasattr(get_generator, "instance"):
        logger.info("创建内容生成器实例")
        
        # 从环境变量或配置文件加载配置
        try:
            # LLM配置
            llm_config = {
                "model": os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
                "api_key": os.getenv("OPENAI_API_KEY", None),
                "api_base": os.getenv("OPENAI_API_BASE", None)
            }
            
            # 内容生成器配置
            quality_threshold = float(os.getenv("GENERATOR_QUALITY_THRESHOLD", "0.75"))
            enable_quality_check = os.getenv("GENERATOR_ENABLE_QUALITY_CHECK", "true").lower() == "true"
            
            # 创建生成器
            get_generator.instance = create_content_generator(
                retriever=retriever,
                llm_config=llm_config,
                quality_threshold=quality_threshold,
                enable_quality_check=enable_quality_check
            )
            
            logger.info("内容生成器创建成功，配置如下:")
            logger.info(f"LLM模型: {llm_config['model']}")
            logger.info(f"质量阈值: {quality_threshold}")
            logger.info(f"质量检查: {'启用' if enable_quality_check else '禁用'}")
        except Exception as e:
            logger.error(f"创建内容生成器失败: {str(e)}", exc_info=True)
            # 创建一个基础的生成器
            get_generator.instance = create_content_generator(retriever=retriever)
            logger.info("已创建基础内容生成器")
    
    return get_generator.instance


# API路由
@app.get("/")
async def root():
    """
    API根路径，返回基本信息
    """
    return {
        "service": "智能教育助手 - 知识服务API",
        "version": "0.1.0",
        "status": "运行中",
        "endpoints": [
            "/api/search - 知识检索",
            "/api/generate - 内容生成",
            "/api/knowledge - 添加知识项"
        ]
    }


@app.post("/api/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    retriever: KnowledgeRetriever = Depends(get_retriever)
) -> SearchResponse:
    """
    知识检索API
    
    根据查询和过滤条件，返回相关的知识项
    """
    logger.info(f"收到检索请求: {request.query}")
    start_time = time.time()
    
    try:
        # 执行检索
        results = retriever.retrieve(
            query=request.query,
            grade_level=request.grade_level,
            subject=request.subject,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        # 转换结果为响应格式
        items = []
        for result in results:
            item_dict = result.item.dict()
            item_dict["score"] = result.score
            item_dict["keyword_score"] = result.keyword_score
            item_dict["vector_score"] = result.vector_score
            items.append(item_dict)
        
        # 计算检索耗时
        search_time = time.time() - start_time
        
        return SearchResponse(
            items=items,
            total=len(items),
            query=request.query,
            search_time=search_time
        )
    
    except Exception as e:
        logger.error(f"检索失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"检索过程发生错误: {str(e)}"
        )


@app.post("/api/generate", response_model=ContentResponse)
async def generate(
    request: ContentRequest,
    generator: ContentGenerator = Depends(get_generator)
) -> ContentResponse:
    """
    内容生成API
    
    根据用户查询和参数，生成符合要求的教育内容
    """
    logger.info(f"收到生成请求: {request.query}")
    start_time = time.time()
    
    try:
        # 转换请求格式
        gen_request = GenerationRequest(
            query=request.query,
            grade_level=request.grade_level,
            subject=request.subject,
            max_length=request.max_length,
            temperature=request.temperature,
            format=request.format,
            style=request.style
        )
        
        # 生成内容
        result = generator.generate(gen_request)
        
        # 转换结果为响应格式
        sources = [source.dict() for source in result.sources]
        
        # 计算生成耗时
        generation_time = time.time() - start_time
        
        return ContentResponse(
            content=result.content,
            sources=sources,
            quality_score=result.quality_score,
            generation_time=generation_time
        )
    
    except Exception as e:
        logger.error(f"生成失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"内容生成过程发生错误: {str(e)}"
        )


@app.post("/api/knowledge", response_model=AddKnowledgeItemResponse)
async def add_knowledge_item(
    request: AddKnowledgeItemRequest,
    retriever: KnowledgeRetriever = Depends(get_retriever)
) -> AddKnowledgeItemResponse:
    """
    添加知识项API
    
    将新的知识项添加到知识库(向量数据库和关键词索引)
    """
    logger.info(f"收到添加知识项请求: {request.title}")
    
    try:
        # 生成唯一ID
        item_id = f"item_{int(time.time() * 1000)}_{hash(request.title) % 10000:04d}"
        
        # 创建知识项
        item = KnowledgeItem(
            id=item_id,
            title=request.title,
            content=request.content,
            grade_level=request.grade_level,
            subject=request.subject,
            keywords=request.keywords,
            source=request.source,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # 获取或生成向量表示
        vector = None
        if retriever.vector_db_client is not None:
            # 生成向量表示
            vector = retriever._get_embedding_vector(request.title + " " + request.content)
            if vector:
                item.vector = vector
        
        # 添加到向量数据库
        vector_db_status = False
        if retriever.vector_db_client is not None and vector is not None:
            try:
                vector_db_status = retriever.vector_db_client.add(item_id, vector)
                logger.info(f"向量数据库添加状态: {vector_db_status}")
            except Exception as e:
                logger.error(f"添加到向量数据库失败: {str(e)}", exc_info=True)
        
        # 添加到关键词索引
        keyword_index_status = False
        if retriever.keyword_index is not None:
            try:
                # 转换为字典
                item_dict = item.dict()
                # 确保keywords是字符串(Elasticsearch要求)
                if isinstance(item_dict["keywords"], list):
                    item_dict["keywords"] = ",".join(item_dict["keywords"])
                # 如果有向量，也存储到ES以便直接检索完整项目
                keyword_index_status = retriever.keyword_index.add(item_dict)
                logger.info(f"关键词索引添加状态: {keyword_index_status}")
            except Exception as e:
                logger.error(f"添加到关键词索引失败: {str(e)}", exc_info=True)
        
        status = "success"
        if not vector_db_status and not keyword_index_status:
            status = "failed"
        elif not vector_db_status or not keyword_index_status:
            status = "partial_success"
        
        return AddKnowledgeItemResponse(
            id=item_id,
            status=status,
            vector_db_status=vector_db_status,
            keyword_index_status=keyword_index_status
        )
    
    except Exception as e:
        logger.error(f"添加知识项失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加知识项过程发生错误: {str(e)}"
        )


# 健康检查端点
@app.get("/health")
async def health_check():
    """
    健康检查API
    
    返回服务健康状态和基本组件信息
    """
    # 获取检索器状态
    retriever = get_retriever()
    vector_db_status = "未配置"
    keyword_index_status = "未配置"
    
    # 检查向量数据库状态
    if retriever.vector_db_client is not None:
        try:
            stats = retriever.vector_db_client.stats()
            vector_db_status = stats
        except Exception as e:
            vector_db_status = f"错误: {str(e)}"
    
    # 检查关键词索引状态
    if retriever.keyword_index is not None:
        try:
            stats = retriever.keyword_index.stats()
            keyword_index_status = stats
        except Exception as e:
            keyword_index_status = f"错误: {str(e)}"
    
    return {
        "status": "健康",
        "time": time.time(),
        "components": {
            "vector_db": vector_db_status,
            "keyword_index": keyword_index_status
        }
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """
    启动API服务器
    
    Args:
        host: 主机地址
        port: 端口号
    """
    logger.info(f"启动知识服务API，地址: {host}:{port}")
    uvicorn.run(app, host=host, port=port)


# 当作为脚本直接运行时，启动服务器
if __name__ == "__main__":
    start_server() 