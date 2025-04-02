#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
课程体系API模块

提供课程体系、知识点和学习路径等相关的RESTful API接口。
"""

import os
import time
import logging
import json
from typing import Dict, List, Optional, Any, Union, Set

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from .knowledge_model import (
    Subject, GradeLevel, KnowledgePoint, KnowledgeRelation, 
    RelationType, Curriculum
)
from .curriculum_service import CurriculumService, create_curriculum_service


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 创建FastAPI应用
app = FastAPI(
    title="智能教育助手 - 课程体系API",
    description="提供课程体系、知识点和学习路径等相关功能",
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
class ServiceResponse(BaseModel):
    """服务响应模型"""
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    data: Optional[Dict[str, Any]] = Field(None, description="数据")


class KnowledgePointCreate(BaseModel):
    """知识点创建请求"""
    name: str = Field(..., description="知识点名称")
    subject: Subject = Field(..., description="所属学科")
    grade_level: GradeLevel = Field(..., description="适用年级")
    description: str = Field("", description="知识点描述")
    keywords: List[str] = Field(default=[], description="关键词")
    difficulty: float = Field(0.5, description="难度系数(0-1)")
    importance: float = Field(0.5, description="重要性(0-1)")
    parent_id: Optional[str] = Field(None, description="父级知识点ID")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")


class KnowledgePointUpdate(BaseModel):
    """知识点更新请求"""
    name: Optional[str] = Field(None, description="知识点名称")
    description: Optional[str] = Field(None, description="知识点描述")
    keywords: Optional[List[str]] = Field(None, description="关键词")
    difficulty: Optional[float] = Field(None, description="难度系数(0-1)")
    importance: Optional[float] = Field(None, description="重要性(0-1)")
    parent_id: Optional[str] = Field(None, description="父级知识点ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    
    @validator('difficulty', 'importance')
    def check_range(cls, v, field):
        """检查数值范围"""
        if v is not None and (v < 0 or v > 1):
            field_name = field.name
            raise ValueError(f"{field_name}必须在0到1之间")
        return v


class KnowledgePointResponse(BaseModel):
    """知识点响应"""
    id: str = Field(..., description="知识点ID")
    name: str = Field(..., description="知识点名称")
    subject: Subject = Field(..., description="所属学科")
    grade_level: GradeLevel = Field(..., description="适用年级")
    description: str = Field(..., description="知识点描述")
    keywords: List[str] = Field(..., description="关键词")
    difficulty: float = Field(..., description="难度系数(0-1)")
    importance: float = Field(..., description="重要性(0-1)")
    parent_id: Optional[str] = Field(None, description="父级知识点ID")
    created_at: float = Field(..., description="创建时间")
    updated_at: float = Field(..., description="更新时间")
    metadata: Dict[str, Any] = Field(..., description="元数据")


class KnowledgeRelationCreate(BaseModel):
    """知识点关系创建请求"""
    source_id: str = Field(..., description="源知识点ID")
    target_id: str = Field(..., description="目标知识点ID")
    relation_type: RelationType = Field(..., description="关系类型")
    weight: float = Field(1.0, description="关系权重")
    description: str = Field("", description="关系描述")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")


class KnowledgeRelationResponse(BaseModel):
    """知识点关系响应"""
    source_id: str = Field(..., description="源知识点ID")
    target_id: str = Field(..., description="目标知识点ID")
    relation_type: RelationType = Field(..., description="关系类型")
    weight: float = Field(..., description="关系权重")
    description: str = Field(..., description="关系描述")
    metadata: Dict[str, Any] = Field(..., description="元数据")


class KnowledgeFilterRequest(BaseModel):
    """知识点过滤请求"""
    subject: Optional[Subject] = Field(None, description="学科")
    grade_level: Optional[GradeLevel] = Field(None, description="年级")
    difficulty_min: float = Field(0.0, description="最小难度")
    difficulty_max: float = Field(1.0, description="最大难度")
    importance_min: float = Field(0.0, description="最小重要性")
    keywords: Optional[List[str]] = Field(None, description="关键词列表")
    
    @validator('difficulty_min', 'difficulty_max', 'importance_min')
    def check_range(cls, v, field):
        """检查数值范围"""
        if v < 0 or v > 1:
            field_name = field.name
            raise ValueError(f"{field_name}必须在0到1之间")
        return v


class LearningPathRequest(BaseModel):
    """学习路径请求"""
    target_kp_id: str = Field(..., description="目标知识点ID")
    known_kp_ids: List[str] = Field(default=[], description="已掌握的知识点ID列表")


class KnowledgeImportRequest(BaseModel):
    """知识导入请求"""
    data: List[Dict[str, Any]] = Field(..., description="知识数据")


# 依赖项：获取课程体系服务
def get_curriculum_service() -> CurriculumService:
    """
    获取课程体系服务实例
    """
    if not hasattr(get_curriculum_service, "instance"):
        # 从环境变量或配置中读取参数
        storage_path = os.getenv("CURRICULUM_STORAGE_PATH", "./data/curriculum")
        
        config = {
            "storage_path": storage_path
        }
        
        get_curriculum_service.instance = create_curriculum_service(config)
        logger.info(f"课程体系服务已创建，存储路径: {storage_path}")
    
    return get_curriculum_service.instance


# API路由
@app.get("/")
async def root():
    """
    API根路径，返回基本信息
    """
    return {
        "service": "智能教育助手 - 课程体系API",
        "version": "0.1.0",
        "status": "运行中",
        "endpoints": [
            "/api/knowledge_points - 知识点管理",
            "/api/knowledge_relations - 知识点关系管理",
            "/api/curriculum - 课程体系管理",
            "/api/learning_path - 学习路径规划",
            "/api/filter - 知识点过滤"
        ]
    }


@app.post("/api/knowledge_points", response_model=ServiceResponse)
async def create_knowledge_point(
    request: KnowledgePointCreate,
    curriculum_service: CurriculumService = Depends(get_curriculum_service)
) -> ServiceResponse:
    """
    创建知识点API
    
    创建新的知识点
    """
    try:
        # 生成知识点ID
        kp_id = KnowledgePoint.create_id(request.subject, request.name)
        
        # 创建知识点对象
        kp = KnowledgePoint(
            id=kp_id,
            name=request.name,
            subject=request.subject,
            grade_level=request.grade_level,
            description=request.description,
            keywords=request.keywords,
            difficulty=request.difficulty,
            importance=request.importance,
            parent_id=request.parent_id,
            metadata=request.metadata
        )
        
        # 添加知识点
        success = curriculum_service.add_knowledge_point(kp)
        
        if not success:
            return ServiceResponse(
                status="error",
                message="创建知识点失败",
                data=None
            )
        
        return ServiceResponse(
            status="success",
            message="创建知识点成功",
            data={"id": kp_id}
        )
    
    except Exception as e:
        logger.error(f"创建知识点失败: {str(e)}")
        return ServiceResponse(
            status="error",
            message=f"创建知识点失败: {str(e)}",
            data=None
        )


@app.get("/api/knowledge_points/{kp_id}", response_model=KnowledgePointResponse)
async def get_knowledge_point(
    kp_id: str = Path(..., description="知识点ID"),
    curriculum_service: CurriculumService = Depends(get_curriculum_service)
) -> KnowledgePointResponse:
    """
    获取知识点API
    
    获取指定知识点的详细信息
    """
    kp = curriculum_service.get_knowledge_point(kp_id)
    
    if not kp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"知识点不存在: {kp_id}"
        )
    
    return KnowledgePointResponse(
        id=kp.id,
        name=kp.name,
        subject=kp.subject,
        grade_level=kp.grade_level,
        description=kp.description,
        keywords=kp.keywords,
        difficulty=kp.difficulty,
        importance=kp.importance,
        parent_id=kp.parent_id,
        created_at=kp.created_at,
        updated_at=kp.updated_at,
        metadata=kp.metadata
    )


@app.put("/api/knowledge_points/{kp_id}", response_model=ServiceResponse)
async def update_knowledge_point(
    request: KnowledgePointUpdate,
    kp_id: str = Path(..., description="知识点ID"),
    curriculum_service: CurriculumService = Depends(get_curriculum_service)
) -> ServiceResponse:
    """
    更新知识点API
    
    更新指定知识点的信息
    """
    # 获取知识点
    kp = curriculum_service.get_knowledge_point(kp_id)
    
    if not kp:
        return ServiceResponse(
            status="error",
            message=f"知识点不存在: {kp_id}",
            data=None
        )
    
    # 更新字段
    if request.name is not None:
        kp.name = request.name
    
    if request.description is not None:
        kp.description = request.description
    
    if request.keywords is not None:
        kp.keywords = request.keywords
    
    if request.difficulty is not None:
        kp.difficulty = request.difficulty
    
    if request.importance is not None:
        kp.importance = request.importance
    
    if request.parent_id is not None:
        kp.parent_id = request.parent_id
    
    if request.metadata is not None:
        kp.metadata = request.metadata
    
    # 更新时间
    kp.updated_at = time.time()
    
    # 更新知识点
    success = curriculum_service.update_knowledge_point(kp)
    
    if not success:
        return ServiceResponse(
            status="error",
            message="更新知识点失败",
            data=None
        )
    
    return ServiceResponse(
        status="success",
        message="更新知识点成功",
        data={"id": kp_id}
    )


@app.delete("/api/knowledge_points/{kp_id}", response_model=ServiceResponse)
async def delete_knowledge_point(
    kp_id: str = Path(..., description="知识点ID"),
    curriculum_service: CurriculumService = Depends(get_curriculum_service)
) -> ServiceResponse:
    """
    删除知识点API
    
    删除指定的知识点
    """
    # 删除知识点
    success = curriculum_service.delete_knowledge_point(kp_id)
    
    if not success:
        return ServiceResponse(
            status="error",
            message=f"删除知识点失败，可能不存在: {kp_id}",
            data=None
        )
    
    return ServiceResponse(
        status="success",
        message="删除知识点成功",
        data={"id": kp_id}
    )


@app.get("/api/knowledge_points", response_model=List[KnowledgePointResponse])
async def list_knowledge_points(
    subject: Optional[Subject] = Query(None, description="学科"),
    grade_level: Optional[GradeLevel] = Query(None, description="年级"),
    curriculum_service: CurriculumService = Depends(get_curriculum_service)
) -> List[KnowledgePointResponse]:
    """
    列出知识点API
    
    列出符合条件的知识点
    """
    # 根据条件获取知识点
    if subject and grade_level:
        kp_list = curriculum_service.get_knowledge_points_by_subject_and_grade(subject, grade_level)
    elif subject:
        kp_list = curriculum_service.get_knowledge_points_by_subject(subject)
    elif grade_level:
        kp_list = curriculum_service.get_knowledge_points_by_grade(grade_level)
    else:
        # 如果没有指定条件，默认返回所有知识点
        kp_list = list(curriculum_service.knowledge_graph.knowledge_points.values())
    
    # 转换为响应模型
    return [
        KnowledgePointResponse(
            id=kp.id,
            name=kp.name,
            subject=kp.subject,
            grade_level=kp.grade_level,
            description=kp.description,
            keywords=kp.keywords,
            difficulty=kp.difficulty,
            importance=kp.importance,
            parent_id=kp.parent_id,
            created_at=kp.created_at,
            updated_at=kp.updated_at,
            metadata=kp.metadata
        )
        for kp in kp_list
    ]


@app.get("/api/knowledge_points/{kp_id}/children", response_model=List[KnowledgePointResponse])
async def get_knowledge_children(
    kp_id: str = Path(..., description="知识点ID"),
    curriculum_service: CurriculumService = Depends(get_curriculum_service)
) -> List[KnowledgePointResponse]:
    """
    获取子知识点API
    
    获取指定知识点的子知识点
    """
    children = curriculum_service.get_knowledge_children(kp_id)
    
    # 转换为响应模型
    return [
        KnowledgePointResponse(
            id=kp.id,
            name=kp.name,
            subject=kp.subject,
            grade_level=kp.grade_level,
            description=kp.description,
            keywords=kp.keywords,
            difficulty=kp.difficulty,
            importance=kp.importance,
            parent_id=kp.parent_id,
            created_at=kp.created_at,
            updated_at=kp.updated_at,
            metadata=kp.metadata
        )
        for kp in children
    ]


@app.get("/api/knowledge_points/{kp_id}/related", response_model=List[KnowledgePointResponse])
async def get_related_knowledge_points(
    kp_id: str = Path(..., description="知识点ID"),
    relation_type: Optional[RelationType] = Query(None, description="关系类型"),
    curriculum_service: CurriculumService = Depends(get_curriculum_service)
) -> List[KnowledgePointResponse]:
    """
    获取相关知识点API
    
    获取与指定知识点相关的知识点
    """
    related_kps = curriculum_service.get_related_knowledge_points(kp_id, relation_type)
    
    # 转换为响应模型
    return [
        KnowledgePointResponse(
            id=kp.id,
            name=kp.name,
            subject=kp.subject,
            grade_level=kp.grade_level,
            description=kp.description,
            keywords=kp.keywords,
            difficulty=kp.difficulty,
            importance=kp.importance,
            parent_id=kp.parent_id,
            created_at=kp.created_at,
            updated_at=kp.updated_at,
            metadata=kp.metadata
        )
        for kp in related_kps
    ]


@app.post("/api/knowledge_relations", response_model=ServiceResponse)
async def create_knowledge_relation(
    request: KnowledgeRelationCreate,
    curriculum_service: CurriculumService = Depends(get_curriculum_service)
) -> ServiceResponse:
    """
    创建知识点关系API
    
    创建新的知识点关系
    """
    try:
        # 创建关系对象
        relation = KnowledgeRelation(
            source_id=request.source_id,
            target_id=request.target_id,
            relation_type=request.relation_type,
            weight=request.weight,
            description=request.description,
            metadata=request.metadata
        )
        
        # 添加关系
        success = curriculum_service.add_knowledge_relation(relation)
        
        if not success:
            return ServiceResponse(
                status="error",
                message="创建知识点关系失败",
                data=None
            )
        
        return ServiceResponse(
            status="success",
            message="创建知识点关系成功",
            data={
                "source_id": request.source_id,
                "target_id": request.target_id,
                "relation_type": request.relation_type
            }
        )
    
    except Exception as e:
        logger.error(f"创建知识点关系失败: {str(e)}")
        return ServiceResponse(
            status="error",
            message=f"创建知识点关系失败: {str(e)}",
            data=None
        )


@app.delete("/api/knowledge_relations", response_model=ServiceResponse)
async def delete_knowledge_relation(
    source_id: str = Query(..., description="源知识点ID"),
    target_id: str = Query(..., description="目标知识点ID"),
    relation_type: RelationType = Query(..., description="关系类型"),
    curriculum_service: CurriculumService = Depends(get_curriculum_service)
) -> ServiceResponse:
    """
    删除知识点关系API
    
    删除指定的知识点关系
    """
    # 删除关系
    success = curriculum_service.delete_knowledge_relation(source_id, target_id, relation_type)
    
    if not success:
        return ServiceResponse(
            status="error",
            message=f"删除知识点关系失败，可能不存在",
            data=None
        )
    
    return ServiceResponse(
        status="success",
        message="删除知识点关系成功",
        data={
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type
        }
    )


@app.post("/api/filter", response_model=List[KnowledgePointResponse])
async def filter_knowledge_points(
    request: KnowledgeFilterRequest,
    curriculum_service: CurriculumService = Depends(get_curriculum_service)
) -> List[KnowledgePointResponse]:
    """
    过滤知识点API
    
    根据条件过滤知识点
    """
    filtered_kps = curriculum_service.filter_knowledge_by_criteria(
        subject=request.subject,
        grade_level=request.grade_level,
        difficulty_min=request.difficulty_min,
        difficulty_max=request.difficulty_max,
        importance_min=request.importance_min,
        keywords=request.keywords
    )
    
    # 转换为响应模型
    return [
        KnowledgePointResponse(
            id=kp.id,
            name=kp.name,
            subject=kp.subject,
            grade_level=kp.grade_level,
            description=kp.description,
            keywords=kp.keywords,
            difficulty=kp.difficulty,
            importance=kp.importance,
            parent_id=kp.parent_id,
            created_at=kp.created_at,
            updated_at=kp.updated_at,
            metadata=kp.metadata
        )
        for kp in filtered_kps
    ]


@app.post("/api/learning_path", response_model=ServiceResponse)
async def plan_learning_path(
    request: LearningPathRequest,
    curriculum_service: CurriculumService = Depends(get_curriculum_service)
) -> ServiceResponse:
    """
    规划学习路径API
    
    规划从已知知识点到目标知识点的学习路径
    """
    # 获取学习路径
    paths = curriculum_service.plan_learning_path(
        target_kp_id=request.target_kp_id,
        known_kp_ids=request.known_kp_ids
    )
    
    if not paths:
        return ServiceResponse(
            status="warning",
            message=f"未找到到达目标知识点的学习路径",
            data={"paths": []}
        )
    
    # 获取知识点详情
    paths_with_details = []
    for path in paths:
        path_details = []
        for kp_id in path:
            kp = curriculum_service.get_knowledge_point(kp_id)
            if kp:
                path_details.append({
                    "id": kp.id,
                    "name": kp.name,
                    "subject": kp.subject,
                    "grade_level": kp.grade_level,
                    "difficulty": kp.difficulty
                })
        paths_with_details.append(path_details)
    
    return ServiceResponse(
        status="success",
        message=f"找到{len(paths)}条学习路径",
        data={
            "target_id": request.target_kp_id,
            "known_ids": request.known_kp_ids,
            "paths": paths_with_details
        }
    )


@app.get("/api/curriculum", response_model=ServiceResponse)
async def get_curriculum_structure(
    subject: Subject = Query(..., description="学科"),
    grade_level: GradeLevel = Query(..., description="年级"),
    curriculum_service: CurriculumService = Depends(get_curriculum_service)
) -> ServiceResponse:
    """
    获取课程结构API
    
    获取指定学科和年级的课程结构
    """
    structure = curriculum_service.get_curriculum_structure(subject, grade_level)
    
    return ServiceResponse(
        status="success",
        message="获取课程结构成功",
        data=structure
    )


@app.post("/api/import", response_model=ServiceResponse)
async def import_knowledge_data(
    request: KnowledgeImportRequest,
    curriculum_service: CurriculumService = Depends(get_curriculum_service)
) -> ServiceResponse:
    """
    导入知识数据API
    
    导入知识点数据
    """
    success_count, fail_count = curriculum_service.import_knowledge_data(request.data)
    
    return ServiceResponse(
        status="success",
        message=f"导入知识数据完成，成功: {success_count}, 失败: {fail_count}",
        data={
            "success_count": success_count,
            "fail_count": fail_count
        }
    )


@app.get("/api/export", response_model=ServiceResponse)
async def export_knowledge_data(
    subject: Optional[Subject] = Query(None, description="学科"),
    curriculum_service: CurriculumService = Depends(get_curriculum_service)
) -> ServiceResponse:
    """
    导出知识数据API
    
    导出知识点数据
    """
    data = curriculum_service.export_knowledge_data(subject)
    
    return ServiceResponse(
        status="success",
        message=f"导出知识数据成功，共{len(data)}个知识点",
        data={"knowledge_data": data}
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


def start_server(host: str = "0.0.0.0", port: int = 8003):
    """
    启动API服务器
    
    Args:
        host: 主机地址
        port: 端口号
    """
    logger.info(f"启动课程体系API，地址: {host}:{port}")
    uvicorn.run(app, host=host, port=port) 