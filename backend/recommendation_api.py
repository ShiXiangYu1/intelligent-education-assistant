#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
个性化推荐API模块

该模块实现了个性化推荐引擎的API接口，包括练习题推荐和学生模型更新功能。
使用FastAPI框架提供RESTful API服务。
"""

import os
import time
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .recommendation_engine import (
    RecommendationEngine, RecommendationRequest, RecommendationResponse,
    PracticeRecord, create_recommendation_engine
)


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 创建FastAPI应用
app = FastAPI(
    title="智能教育助手 - 个性化推荐API",
    description="提供个性化练习题推荐和学生模型更新功能的API接口",
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
class RecommendRequest(BaseModel):
    """推荐请求模型"""
    student_id: str = Field(..., description="学生ID")
    subject: Optional[str] = Field(None, description="学科")
    knowledge_points: Optional[List[str]] = Field(None, description="指定知识点列表")
    count: int = Field(5, description="推荐数量")
    difficulty_range: Optional[Tuple[float, float]] = Field(None, description="难度范围(0-1)")
    exclude_practiced: bool = Field(False, description="是否排除已练习题目")
    practice_type: Optional[str] = Field(None, description="练习类型")
    priority: str = Field("balanced", description="优先策略(weak_first/reinforce/balanced)")


class PracticeRecordRequest(BaseModel):
    """练习记录请求模型"""
    id: Optional[str] = Field(None, description="记录ID")
    student_id: str = Field(..., description="学生ID")
    exercise_id: str = Field(..., description="练习题ID")
    knowledge_points: List[str] = Field(..., description="相关知识点ID列表")
    is_correct: bool = Field(..., description="是否正确")
    answer: str = Field(..., description="学生答案")
    time_spent: float = Field(..., description="耗时(秒)")
    difficulty: Optional[float] = Field(None, description="难度系数(0-1)")
    feedback: Optional[str] = Field(None, description="反馈")


class PracticeRecordResponse(BaseModel):
    """练习记录响应模型"""
    id: str = Field(..., description="记录ID")
    student_id: str = Field(..., description="学生ID")
    status: str = Field(..., description="状态")
    knowledge_updates: Dict[str, float] = Field(..., description="知识点掌握度更新")
    processed_time: float = Field(..., description="处理时间")


# 依赖项：获取推荐引擎实例
def get_recommendation_engine() -> RecommendationEngine:
    """
    获取推荐引擎实例
    
    这里使用单例模式，确保应用中只创建一个推荐引擎实例
    """
    # 实际项目中，这里应该从配置中读取参数
    if not hasattr(get_recommendation_engine, "instance"):
        logger.info("创建推荐引擎实例")
        
        # 从环境变量或配置文件加载配置
        try:
            # 配置参数
            base_retention = float(os.getenv("FORGETTING_BASE_RETENTION", "0.9"))
            forgetting_rate = float(os.getenv("FORGETTING_RATE", "0.1"))
            practice_boost = float(os.getenv("PRACTICE_BOOST", "0.1"))
            
            # 创建推荐引擎
            get_recommendation_engine.instance = create_recommendation_engine()
            
            logger.info("推荐引擎创建成功，配置如下:")
            logger.info(f"基础记忆保持率: {base_retention}")
            logger.info(f"遗忘速率: {forgetting_rate}")
            logger.info(f"练习提升: {practice_boost}")
        except Exception as e:
            logger.error(f"创建推荐引擎失败: {str(e)}", exc_info=True)
            # 创建一个基础的推荐引擎
            get_recommendation_engine.instance = create_recommendation_engine()
            logger.info("已创建基础推荐引擎")
    
    return get_recommendation_engine.instance


# API路由
@app.get("/")
async def root():
    """
    API根路径，返回基本信息
    """
    return {
        "service": "智能教育助手 - 个性化推荐API",
        "version": "0.1.0",
        "status": "运行中",
        "endpoints": [
            "/api/recommend - 练习题推荐",
            "/api/practice - 提交练习记录",
            "/api/health - 健康检查"
        ]
    }


@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend(
    request: RecommendRequest,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
) -> RecommendationResponse:
    """
    练习题推荐API
    
    根据学生情况和请求参数，推荐个性化练习题
    """
    logger.info(f"收到推荐请求 - 学生: {request.student_id}, 学科: {request.subject}")
    
    try:
        # 准备推荐请求
        difficulty_range = request.difficulty_range or (0.0, 1.0)
        
        recommendation_request = RecommendationRequest(
            student_id=request.student_id,
            subject=request.subject,
            knowledge_points=request.knowledge_points,
            count=request.count,
            difficulty_range=difficulty_range,
            exclude_practiced=request.exclude_practiced,
            practice_type=request.practice_type,
            priority=request.priority
        )
        
        # 执行推荐
        result = engine.recommend(recommendation_request)
        
        logger.info(f"推荐完成 - 学生: {request.student_id}, 推荐数量: {len(result.recommendations)}")
        return result
    
    except Exception as e:
        logger.error(f"推荐失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"推荐过程发生错误: {str(e)}"
        )


@app.post("/api/practice", response_model=PracticeRecordResponse)
async def submit_practice(
    request: PracticeRecordRequest,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
) -> PracticeRecordResponse:
    """
    提交练习记录API
    
    记录学生的练习情况，并更新学生模型
    """
    logger.info(f"收到练习记录 - 学生: {request.student_id}, 练习: {request.exercise_id}")
    
    try:
        # 生成记录ID(如果未提供)
        record_id = request.id or f"record_{int(time.time() * 1000)}_{hash(request.student_id + request.exercise_id) % 10000:04d}"
        
        # 准备练习记录
        practice_record = PracticeRecord(
            id=record_id,
            student_id=request.student_id,
            exercise_id=request.exercise_id,
            knowledge_points=request.knowledge_points,
            is_correct=request.is_correct,
            answer=request.answer,
            time_spent=request.time_spent,
            practice_time=time.time(),
            difficulty=request.difficulty or 0.5,
            feedback=request.feedback
        )
        
        # 更新学生模型
        updated_student = engine.update_student_model(
            student_id=request.student_id,
            practice_record=practice_record
        )
        
        # 准备响应
        knowledge_updates = {}
        if updated_student:
            for kp_id in request.knowledge_points:
                if kp_id in updated_student.knowledge_mastery:
                    knowledge_updates[kp_id] = updated_student.knowledge_mastery[kp_id].mastery_level
        
        return PracticeRecordResponse(
            id=record_id,
            student_id=request.student_id,
            status="success" if updated_student else "partial_success",
            knowledge_updates=knowledge_updates,
            processed_time=time.time()
        )
    
    except Exception as e:
        logger.error(f"处理练习记录失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理练习记录过程发生错误: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    健康检查API
    
    返回服务健康状态
    """
    return {
        "status": "健康",
        "time": time.time()
    }


def start_server(host: str = "0.0.0.0", port: int = 8001):
    """
    启动API服务器
    
    Args:
        host: 主机地址
        port: 端口号
    """
    logger.info(f"启动个性化推荐API，地址: {host}:{port}")
    uvicorn.run(app, host=host, port=port) 