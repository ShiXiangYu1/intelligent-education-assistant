#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用户服务模块

该模块提供用户管理、认证和学习记录相关功能。
包含用户注册、登录、个人资料管理、学习进度追踪和用户画像与学习记录整合等功能。
"""

from .user_model import User, UserProfile, UserRole, LearningRecord
from .auth import create_auth_manager, AuthManager, Token, LoginRequest, LoginResponse
from .user_service import create_user_service, UserService
from .user_learning_integration import (
    UserLearningIntegration, LearningStats, UserLearningConfig,
    create_user_learning_integration
)

__all__ = [
    'User', 'UserProfile', 'UserRole', 'LearningRecord',
    'AuthManager', 'Token', 'LoginRequest', 'LoginResponse', 'create_auth_manager',
    'UserService', 'create_user_service',
    'UserLearningIntegration', 'LearningStats', 'UserLearningConfig', 'create_user_learning_integration',
] 