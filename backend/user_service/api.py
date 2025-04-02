#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用户服务API模块

提供用户管理相关的RESTful API接口，包括用户注册、登录、退出、
用户信息查询和更新、学习记录分析与整合等功能。
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr

from .user_model import User, UserProfile, UserRole
from .auth import (
    AuthManager, Token, LoginRequest, LoginResponse, 
    RegisterRequest, create_auth_manager
)
from .user_service import UserService, create_user_service
from .user_learning_integration import (
    UserLearningIntegration, LearningStats, UserLearningConfig,
    create_user_learning_integration
)

# 导入推荐引擎模块
from backend.recommendation_engine import RecommendationEngine, create_recommendation_engine


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 创建FastAPI应用
app = FastAPI(
    title="智能教育助手 - 用户服务API",
    description="提供用户注册、登录、退出和用户信息管理功能",
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

# 设置OAuth2认证
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# API请求和响应模型
class UserResponse(BaseModel):
    """用户响应模型"""
    id: str = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    email: str = Field(..., description="电子邮箱")
    role: UserRole = Field(..., description="用户角色")
    real_name: Optional[str] = Field(None, description="真实姓名")
    avatar: Optional[str] = Field(None, description="头像URL")
    grade_level: Optional[int] = Field(None, description="年级")
    subjects: List[str] = Field(default=[], description="关注的学科")
    is_active: bool = Field(..., description="是否激活")
    created_at: float = Field(..., description="创建时间")
    last_login: Optional[float] = Field(None, description="最后登录时间")


class UpdateUserRequest(BaseModel):
    """更新用户请求模型"""
    real_name: Optional[str] = Field(None, description="真实姓名")
    email: Optional[EmailStr] = Field(None, description="电子邮箱")
    phone: Optional[str] = Field(None, description="手机号码")
    avatar: Optional[str] = Field(None, description="头像URL")
    grade_level: Optional[int] = Field(None, description="年级")
    subjects: Optional[List[str]] = Field(None, description="关注的学科")


class UpdatePasswordRequest(BaseModel):
    """更新密码请求模型"""
    old_password: str = Field(..., description="旧密码")
    new_password: str = Field(..., description="新密码")
    confirm_password: str = Field(..., description="确认新密码")


class ServiceResponse(BaseModel):
    """服务响应模型"""
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    data: Optional[Dict[str, Any]] = Field(None, description="数据")


# 依赖项：获取认证管理器
def get_auth_manager() -> AuthManager:
    """
    获取认证管理器实例
    """
    if not hasattr(get_auth_manager, "instance"):
        # 从环境变量或配置中读取参数
        token_secret_key = os.getenv("AUTH_SECRET_KEY", "your-secret-key-please-change-in-production")
        token_expire_minutes = int(os.getenv("TOKEN_EXPIRE_MINUTES", "60"))
        
        config = {
            "token_secret_key": token_secret_key,
            "access_token_expire_minutes": token_expire_minutes
        }
        
        get_auth_manager.instance = create_auth_manager(config)
        logger.info("认证管理器已创建")
    
    return get_auth_manager.instance


# 依赖项：获取用户服务
def get_user_service() -> UserService:
    """
    获取用户服务实例
    """
    if not hasattr(get_user_service, "instance"):
        # 从环境变量或配置中读取参数
        storage_path = os.getenv("USER_STORAGE_PATH", "./data/users")
        
        config = {
            "storage_path": storage_path
        }
        
        get_user_service.instance = create_user_service(config)
        logger.info(f"用户服务已创建，存储路径: {storage_path}")
        
        # 设置用户服务到认证管理器
        auth_manager = get_auth_manager()
        auth_manager.set_user_service(get_user_service.instance)
    
    return get_user_service.instance


# 依赖项：获取当前用户
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    auth_manager: AuthManager = Depends(get_auth_manager),
    user_service: UserService = Depends(get_user_service)
) -> User:
    """
    根据令牌获取当前用户
    
    Args:
        token: 访问令牌
        auth_manager: 认证管理器
        user_service: 用户服务
        
    Returns:
        当前用户对象
        
    Raises:
        HTTPException: 如果认证失败
    """
    payload = auth_manager.decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的身份验证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的身份验证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = user_service.find_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户已被禁用",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


# 依赖项：验证管理员权限
def verify_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    验证当前用户是否具有管理员权限
    
    Args:
        current_user: 当前用户
        
    Returns:
        当前用户
        
    Raises:
        HTTPException: 如果用户不是管理员
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限",
        )
    return current_user


# API路由
@app.get("/")
async def root():
    """
    API根路径，返回基本信息
    """
    return {
        "service": "智能教育助手 - 用户服务API",
        "version": "0.1.0",
        "status": "运行中",
        "endpoints": [
            "/api/auth/register - 用户注册",
            "/api/auth/login - 用户登录",
            "/api/users/me - 获取当前用户信息",
            "/api/users/{user_id} - 获取/更新用户信息"
        ]
    }


@app.post("/api/auth/register", response_model=ServiceResponse)
async def register(
    request: RegisterRequest,
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> ServiceResponse:
    """
    用户注册API
    
    用户注册新账号
    """
    logger.info(f"收到注册请求: {request.username}")
    
    try:
        user, message = auth_manager.register(request)
        
        if not user:
            return ServiceResponse(
                status="error",
                message=message,
                data=None
            )
        
        return ServiceResponse(
            status="success",
            message="注册成功",
            data={"user_id": user.id}
        )
    
    except Exception as e:
        logger.error(f"注册失败: {str(e)}", exc_info=True)
        return ServiceResponse(
            status="error",
            message=f"注册失败: {str(e)}",
            data=None
        )


@app.post("/api/auth/login", response_model=ServiceResponse)
async def login(
    request: LoginRequest,
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> ServiceResponse:
    """
    用户登录API
    
    用户登录并获取访问令牌
    """
    logger.info(f"收到登录请求: {request.username}")
    
    try:
        login_result = auth_manager.login(request)
        
        if not login_result:
            return ServiceResponse(
                status="error",
                message="用户名或密码错误",
                data=None
            )
        
        return ServiceResponse(
            status="success",
            message="登录成功",
            data={
                "token": login_result.token.dict(),
                "user": login_result.user
            }
        )
    
    except Exception as e:
        logger.error(f"登录失败: {str(e)}", exc_info=True)
        return ServiceResponse(
            status="error",
            message=f"登录失败: {str(e)}",
            data=None
        )


@app.get("/api/users/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
) -> UserResponse:
    """
    获取当前用户信息API
    
    获取当前登录用户的详细信息
    """
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        real_name=current_user.real_name,
        avatar=current_user.avatar,
        grade_level=current_user.profile.grade_level if current_user.profile else None,
        subjects=current_user.profile.subjects if current_user.profile else [],
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )


@app.get("/api/users/{user_id}", response_model=UserResponse)
async def get_user_info(
    user_id: str = Path(..., description="用户ID"),
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
) -> UserResponse:
    """
    获取用户信息API
    
    获取指定用户的详细信息，需要管理员权限或者是获取自己的信息
    """
    # 检查权限
    if current_user.id != user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权访问其他用户信息",
        )
    
    # 获取用户
    user = user_service.find_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户不存在: {user_id}",
        )
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        real_name=user.real_name,
        avatar=user.avatar,
        grade_level=user.profile.grade_level if user.profile else None,
        subjects=user.profile.subjects if user.profile else [],
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login
    )


@app.put("/api/users/{user_id}", response_model=ServiceResponse)
async def update_user_info(
    request: UpdateUserRequest,
    user_id: str = Path(..., description="用户ID"),
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service)
) -> ServiceResponse:
    """
    更新用户信息API
    
    更新指定用户的信息，需要管理员权限或者是更新自己的信息
    """
    # 检查权限
    if current_user.id != user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权修改其他用户信息",
        )
    
    # 获取用户
    user = user_service.find_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户不存在: {user_id}",
        )
    
    # 更新用户信息
    if request.real_name is not None:
        user.real_name = request.real_name
    
    if request.email is not None:
        user.email = request.email
    
    if request.phone is not None:
        user.phone = request.phone
    
    if request.avatar is not None:
        user.avatar = request.avatar
    
    # 更新用户画像
    if not user.profile:
        user.profile = UserProfile()
    
    if request.grade_level is not None:
        user.profile.grade_level = request.grade_level
    
    if request.subjects is not None:
        user.profile.subjects = request.subjects
    
    # 保存更新
    success = user_service.update(user)
    
    if not success:
        return ServiceResponse(
            status="error",
            message="更新用户信息失败",
            data=None
        )
    
    return ServiceResponse(
        status="success",
        message="更新用户信息成功",
        data={"user_id": user.id}
    )


@app.put("/api/users/{user_id}/password", response_model=ServiceResponse)
async def update_password(
    request: UpdatePasswordRequest,
    user_id: str = Path(..., description="用户ID"),
    current_user: User = Depends(get_current_user),
    user_service: UserService = Depends(get_user_service),
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> ServiceResponse:
    """
    更新密码API
    
    更新指定用户的密码，需要提供旧密码进行验证
    """
    # 检查权限
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权修改其他用户密码",
        )
    
    # 检查两次输入的新密码是否一致
    if request.new_password != request.confirm_password:
        return ServiceResponse(
            status="error",
            message="两次输入的新密码不一致",
            data=None
        )
    
    # 验证旧密码
    if not auth_manager.verify_password(
        request.old_password, 
        current_user.hashed_password, 
        current_user.salt
    ):
        return ServiceResponse(
            status="error",
            message="旧密码验证失败",
            data=None
        )
    
    # 设置新密码
    hashed_password, salt = auth_manager.hash_password(request.new_password)
    current_user.hashed_password = hashed_password
    current_user.salt = salt
    
    # 保存更新
    success = user_service.update(current_user)
    
    if not success:
        return ServiceResponse(
            status="error",
            message="更新密码失败",
            data=None
        )
    
    return ServiceResponse(
        status="success",
        message="更新密码成功",
        data=None
    )


@app.get("/api/users", response_model=List[UserResponse])
async def list_users(
    role: Optional[UserRole] = Query(None, description="用户角色"),
    skip: int = Query(0, description="跳过数量"),
    limit: int = Query(10, description="限制数量"),
    current_user: User = Depends(verify_admin),
    user_service: UserService = Depends(get_user_service)
) -> List[UserResponse]:
    """
    列出用户API
    
    列出系统中的用户，需要管理员权限
    """
    # 根据角色筛选用户
    if role:
        users = user_service.find_by_role(role)
    else:
        users = list(user_service.users_cache.values())
    
    # 分页
    users = users[skip:skip + limit]
    
    # 转换为响应模型
    return [
        UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            real_name=user.real_name,
            avatar=user.avatar,
            grade_level=user.profile.grade_level if user.profile else None,
            subjects=user.profile.subjects if user.profile else [],
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login
        )
        for user in users
    ]


@app.post("/api/users/{user_id}/active", response_model=ServiceResponse)
async def set_user_active(
    user_id: str = Path(..., description="用户ID"),
    active: bool = Query(..., description="是否激活"),
    current_user: User = Depends(verify_admin),
    user_service: UserService = Depends(get_user_service)
) -> ServiceResponse:
    """
    设置用户激活状态API
    
    设置指定用户的激活状态，需要管理员权限
    """
    # 获取用户
    user = user_service.find_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户不存在: {user_id}",
        )
    
    # 更新状态
    user.is_active = active
    
    # 保存更新
    success = user_service.update(user)
    
    if not success:
        return ServiceResponse(
            status="error",
            message="更新用户状态失败",
            data=None
        )
    
    return ServiceResponse(
        status="success",
        message=f"用户已{'激活' if active else '禁用'}",
        data={"user_id": user.id, "is_active": active}
    )


@app.delete("/api/users/{user_id}", response_model=ServiceResponse)
async def delete_user(
    user_id: str = Path(..., description="用户ID"),
    current_user: User = Depends(verify_admin),
    user_service: UserService = Depends(get_user_service)
) -> ServiceResponse:
    """
    删除用户API
    
    删除指定用户，需要管理员权限
    """
    # 检查是否是自己
    if current_user.id == user_id:
        return ServiceResponse(
            status="error",
            message="不能删除自己的账号",
            data=None
        )
    
    # 删除用户
    success = user_service.delete(user_id)
    
    if not success:
        return ServiceResponse(
            status="error",
            message="删除用户失败",
            data=None
        )
    
    return ServiceResponse(
        status="success",
        message="删除用户成功",
        data={"user_id": user_id}
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


# 学习记录相关的请求和响应模型
class PracticeRecordRequest(BaseModel):
    """练习记录请求模型"""
    exercise_id: str = Field(..., description="练习题ID")
    is_correct: bool = Field(..., description="是否正确")
    answer: str = Field(..., description="学生答案")
    time_spent: float = Field(..., description="耗时(秒)")
    knowledge_points: Optional[List[str]] = Field(None, description="相关知识点ID列表")


# 依赖项：获取推荐引擎
def get_recommendation_engine() -> RecommendationEngine:
    """
    获取推荐引擎实例
    """
    if not hasattr(get_recommendation_engine, "instance"):
        # 从环境变量或配置中读取参数
        # 这里简单初始化一个基础的推荐引擎
        get_recommendation_engine.instance = create_recommendation_engine()
        logger.info("推荐引擎已创建")
    
    return get_recommendation_engine.instance


# 依赖项：获取用户学习整合服务
def get_user_learning_integration() -> UserLearningIntegration:
    """
    获取用户学习整合实例
    """
    if not hasattr(get_user_learning_integration, "instance"):
        # 获取用户服务和推荐引擎
        user_service = get_user_service()
        recommendation_engine = get_recommendation_engine()
        
        # 从环境变量或配置中读取参数
        config = {
            "sync_interval": int(os.getenv("LEARNING_SYNC_INTERVAL", "3600")),
            "learning_trend_days": int(os.getenv("LEARNING_TREND_DAYS", "30")),
            "enable_auto_sync": os.getenv("ENABLE_AUTO_SYNC", "true").lower() == "true"
        }
        
        get_user_learning_integration.instance = create_user_learning_integration(
            user_service=user_service,
            recommendation_engine=recommendation_engine,
            config=config
        )
        logger.info("用户学习整合服务已创建")
    
    return get_user_learning_integration.instance


# 学习记录相关的API接口
@app.post("/api/users/{user_id}/practice", response_model=ServiceResponse)
async def submit_practice_record(
    request: PracticeRecordRequest,
    user_id: str = Path(..., description="用户ID"),
    current_user: User = Depends(get_current_user),
    integration: UserLearningIntegration = Depends(get_user_learning_integration)
) -> ServiceResponse:
    """
    提交练习记录
    
    Args:
        request: 练习记录请求
        user_id: 用户ID
        current_user: 当前用户
        integration: 用户学习整合服务
        
    Returns:
        服务响应
    """
    # 检查权限(只能提交自己的或者是管理员)
    if current_user.id != user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限提交其他用户的练习记录"
        )
    
    # 处理练习记录
    success, message = integration.process_practice_record(
        user_id=user_id,
        exercise_id=request.exercise_id,
        is_correct=request.is_correct,
        answer=request.answer,
        time_spent=request.time_spent,
        knowledge_points=request.knowledge_points
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return ServiceResponse(
        status="success",
        message=message,
        data={"user_id": user_id, "exercise_id": request.exercise_id}
    )


@app.get("/api/users/{user_id}/learning/stats", response_model=ServiceResponse)
async def get_user_learning_stats(
    user_id: str = Path(..., description="用户ID"),
    refresh: bool = Query(False, description="是否刷新缓存"),
    current_user: User = Depends(get_current_user),
    integration: UserLearningIntegration = Depends(get_user_learning_integration)
) -> ServiceResponse:
    """
    获取用户学习统计数据
    
    Args:
        user_id: 用户ID
        refresh: 是否刷新缓存
        current_user: 当前用户
        integration: 用户学习整合服务
        
    Returns:
        用户学习统计响应
    """
    # 检查权限(只能查看自己的或者是管理员)
    if current_user.id != user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限查看其他用户的学习统计"
        )
    
    # 获取学习统计
    stats = integration.get_learning_stats(user_id, refresh)
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到用户 {user_id} 的学习统计数据"
        )
    
    return ServiceResponse(
        status="success",
        message="获取学习统计成功",
        data=stats.dict()
    )


@app.get("/api/users/{user_id}/learning/knowledge_mastery", response_model=ServiceResponse)
async def get_knowledge_mastery_detail(
    user_id: str = Path(..., description="用户ID"),
    subject: Optional[str] = Query(None, description="学科"),
    current_user: User = Depends(get_current_user),
    integration: UserLearningIntegration = Depends(get_user_learning_integration)
) -> ServiceResponse:
    """
    获取用户知识点掌握详情
    
    Args:
        user_id: 用户ID
        subject: 学科(可选)
        current_user: 当前用户
        integration: 用户学习整合服务
        
    Returns:
        知识点掌握详情响应
    """
    # 检查权限(只能查看自己的或者是管理员)
    if current_user.id != user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限查看其他用户的知识点掌握详情"
        )
    
    # 获取知识点掌握详情
    mastery_detail = integration.get_knowledge_mastery_detail(user_id, subject)
    
    return ServiceResponse(
        status="success",
        message="获取知识点掌握详情成功",
        data={"knowledge_points": mastery_detail}
    )


@app.get("/api/users/{user_id}/learning/path", response_model=ServiceResponse)
async def get_recommended_learning_path(
    user_id: str = Path(..., description="用户ID"),
    subject: str = Query(..., description="学科"),
    target_kp: Optional[str] = Query(None, description="目标知识点"),
    current_user: User = Depends(get_current_user),
    integration: UserLearningIntegration = Depends(get_user_learning_integration)
) -> ServiceResponse:
    """
    获取推荐学习路径
    
    Args:
        user_id: 用户ID
        subject: 学科
        target_kp: 目标知识点(可选)
        current_user: 当前用户
        integration: 用户学习整合服务
        
    Returns:
        学习路径响应
    """
    # 检查权限(只能查看自己的或者是管理员)
    if current_user.id != user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限查看其他用户的推荐学习路径"
        )
    
    # 获取推荐学习路径
    paths = integration.get_recommended_learning_path(
        user_id=user_id,
        subject=subject,
        target_knowledge_point=target_kp
    )
    
    return ServiceResponse(
        status="success",
        message="获取推荐学习路径成功",
        data={"paths": paths}
    )


@app.post("/api/users/{user_id}/learning/sync", response_model=ServiceResponse)
async def sync_user_learning(
    user_id: str = Path(..., description="用户ID"),
    current_user: User = Depends(get_current_user),
    integration: UserLearningIntegration = Depends(get_user_learning_integration)
) -> ServiceResponse:
    """
    同步用户学习记录
    
    Args:
        user_id: 用户ID
        current_user: 当前用户
        integration: 用户学习整合服务
        
    Returns:
        服务响应
    """
    # 检查权限(只能同步自己的或者是管理员)
    if current_user.id != user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限同步其他用户的学习记录"
        )
    
    # 同步到学生模型
    to_student_success, to_student_message = integration.sync_user_to_student_model(user_id)
    
    # 同步到用户画像
    to_user_success, to_user_message = integration.sync_learning_records_to_user(user_id)
    
    if not to_student_success and not to_user_success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"同步失败: {to_student_message}, {to_user_message}"
        )
    
    return ServiceResponse(
        status="success",
        message="学习记录同步完成",
        data={
            "user_id": user_id,
            "to_student": {"success": to_student_success, "message": to_student_message},
            "to_user": {"success": to_user_success, "message": to_user_message}
        }
    )


def start_server(host: str = "0.0.0.0", port: int = 8002):
    """
    启动API服务器
    
    Args:
        host: 主机地址
        port: 端口号
    """
    logger.info(f"启动用户服务API，地址: {host}:{port}")
    uvicorn.run(app, host=host, port=port) 