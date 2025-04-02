#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
认证管理模块

提供用户认证、令牌管理和密码处理相关功能。
"""

import os
import time
import uuid
import hashlib
import logging
import secrets
import base64
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta

import jwt
from pydantic import BaseModel, Field, EmailStr, validator

from .user_model import User, UserRole


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 数据模型
class Token(BaseModel):
    """令牌模型"""
    access_token: str = Field(..., description="访问令牌")
    token_type: str = Field("bearer", description="令牌类型")
    expires_at: float = Field(..., description="过期时间")
    user_id: str = Field(..., description="用户ID")
    role: UserRole = Field(..., description="用户角色")


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")
    remember_me: bool = Field(False, description="记住我")


class LoginResponse(BaseModel):
    """登录响应"""
    token: Token = Field(..., description="认证令牌")
    user: Dict[str, Any] = Field(..., description="用户信息")


class RegisterRequest(BaseModel):
    """注册请求"""
    username: str = Field(..., description="用户名")
    email: EmailStr = Field(..., description="电子邮箱")
    password: str = Field(..., description="密码")
    password_confirm: str = Field(..., description="确认密码")
    role: UserRole = Field(UserRole.STUDENT, description="用户角色")
    
    @validator('password')
    def password_strength(cls, v):
        """验证密码强度"""
        if len(v) < 8:
            raise ValueError('密码长度必须至少为8个字符')
        if not any(c.isdigit() for c in v):
            raise ValueError('密码必须包含至少一个数字')
        if not any(c.isalpha() for c in v):
            raise ValueError('密码必须包含至少一个字母')
        return v
    
    @validator('password_confirm')
    def passwords_match(cls, v, values):
        """验证两次密码输入是否一致"""
        if 'password' in values and v != values['password']:
            raise ValueError('两次输入的密码不一致')
        return v


class AuthConfig(BaseModel):
    """认证配置"""
    token_secret_key: str = Field(..., description="令牌密钥")
    token_algorithm: str = Field("HS256", description="令牌算法")
    access_token_expire_minutes: int = Field(60, description="访问令牌过期时间(分钟)")
    refresh_token_expire_days: int = Field(7, description="刷新令牌过期时间(天)")
    password_salt_size: int = Field(32, description="密码盐长度")


class AuthManager:
    """认证管理器"""
    
    def __init__(self, config: AuthConfig, user_service=None):
        """
        初始化认证管理器
        
        Args:
            config: 认证配置
            user_service: 用户服务实例
        """
        self.config = config
        self.user_service = user_service
        logger.info("初始化认证管理器")
    
    def set_user_service(self, user_service):
        """设置用户服务实例"""
        self.user_service = user_service
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        密码哈希
        
        Args:
            password: 明文密码
            salt: 盐值，如果为None则自动生成
            
        Returns:
            (hashed_password, salt) 哈希后的密码和盐值
        """
        if salt is None:
            salt = secrets.token_hex(self.config.password_salt_size)
        
        # 使用PBKDF2算法计算哈希
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 迭代次数
        )
        
        hashed_password = base64.b64encode(key).decode('utf-8')
        return hashed_password, salt
    
    def verify_password(self, plain_password: str, hashed_password: str, salt: str) -> bool:
        """
        验证密码
        
        Args:
            plain_password: 明文密码
            hashed_password: 哈希后的密码
            salt: 盐值
            
        Returns:
            密码是否正确
        """
        calculated_hash, _ = self.hash_password(plain_password, salt)
        return calculated_hash == hashed_password
    
    def create_token(self, user_id: str, role: UserRole, remember_me: bool = False) -> Token:
        """
        创建访问令牌
        
        Args:
            user_id: 用户ID
            role: 用户角色
            remember_me: 是否记住我
            
        Returns:
            Token对象
        """
        expire_minutes = self.config.access_token_expire_minutes
        if remember_me:
            expire_minutes = self.config.access_token_expire_minutes * 24 * 7  # 7天
        
        expires_delta = timedelta(minutes=expire_minutes)
        expire_timestamp = time.time() + expires_delta.total_seconds()
        
        # 创建JWT令牌
        to_encode = {
            "sub": user_id,
            "role": role,
            "exp": expire_timestamp,
            "jti": str(uuid.uuid4())
        }
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.config.token_secret_key,
            algorithm=self.config.token_algorithm
        )
        
        # 返回Token对象
        return Token(
            access_token=encoded_jwt,
            token_type="bearer",
            expires_at=expire_timestamp,
            user_id=user_id,
            role=role
        )
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        解码令牌
        
        Args:
            token: JWT令牌
            
        Returns:
            解码后的令牌数据，或None表示令牌无效
        """
        try:
            payload = jwt.decode(
                token,
                self.config.token_secret_key,
                algorithms=[self.config.token_algorithm]
            )
            return payload
        except jwt.PyJWTError as e:
            logger.warning(f"解码令牌失败: {str(e)}")
            return None
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        用户认证
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            认证成功返回User对象，否则返回None
        """
        if not self.user_service:
            logger.error("用户服务未设置，无法完成认证")
            return None
        
        # 根据用户名查找用户
        user = self.user_service.find_by_username(username)
        if not user:
            logger.warning(f"用户名不存在: {username}")
            return None
        
        # 验证密码
        if not self.verify_password(password, user.hashed_password, user.salt):
            logger.warning(f"密码验证失败: {username}")
            return None
        
        # 更新最后登录时间
        user.update_last_login()
        self.user_service.update(user)
        
        return user
    
    def login(self, request: LoginRequest) -> Optional[LoginResponse]:
        """
        用户登录
        
        Args:
            request: 登录请求
            
        Returns:
            登录成功返回LoginResponse对象，否则返回None
        """
        # 认证用户
        user = self.authenticate(request.username, request.password)
        if not user:
            return None
        
        # 创建令牌
        token = self.create_token(user.id, user.role, request.remember_me)
        
        # 返回登录响应
        return LoginResponse(
            token=token,
            user=user.to_public_dict()
        )
    
    def register(self, request: RegisterRequest) -> Tuple[Optional[User], str]:
        """
        用户注册
        
        Args:
            request: 注册请求
            
        Returns:
            (User, message) 注册成功返回User对象和成功消息，否则返回None和错误消息
        """
        if not self.user_service:
            logger.error("用户服务未设置，无法完成注册")
            return None, "服务器内部错误"
        
        # 检查用户名是否已存在
        if self.user_service.find_by_username(request.username):
            logger.warning(f"用户名已存在: {request.username}")
            return None, "用户名已被使用"
        
        # 检查邮箱是否已存在
        if self.user_service.find_by_email(request.email):
            logger.warning(f"邮箱已存在: {request.email}")
            return None, "邮箱已被注册"
        
        # 哈希密码
        hashed_password, salt = self.hash_password(request.password)
        
        # 创建用户ID
        user_id = f"user_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # 创建用户对象
        user = User(
            id=user_id,
            username=request.username,
            email=request.email,
            hashed_password=hashed_password,
            salt=salt,
            role=request.role,
            is_active=True,
            is_verified=False,
            created_at=time.time()
        )
        
        # 保存用户
        success = self.user_service.create(user)
        if not success:
            logger.error(f"保存用户失败: {request.username}")
            return None, "创建用户失败，请稍后再试"
        
        logger.info(f"用户注册成功: {user_id}, 用户名: {request.username}")
        return user, "注册成功"


def create_auth_manager(
    config: Dict = None,
    user_service = None
) -> AuthManager:
    """
    创建认证管理器
    
    Args:
        config: 配置字典
        user_service: 用户服务实例
        
    Returns:
        认证管理器实例
    """
    if config is None:
        config = {}
    
    # 设置默认密钥，生产环境应当设置安全的密钥
    if 'token_secret_key' not in config:
        config['token_secret_key'] = os.getenv(
            'AUTH_SECRET_KEY',
            secrets.token_hex(32)  # 生成一个随机密钥
        )
    
    auth_config = AuthConfig(**config)
    return AuthManager(auth_config, user_service) 