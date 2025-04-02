#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用户服务模块

提供用户管理和数据持久化相关功能，包括用户创建、查找、更新和删除等操作。
"""

import os
import time
import json
import logging
import threading
from typing import Dict, List, Optional, Set, Union, Any

from pydantic import BaseModel, Field

from .user_model import User, UserProfile, UserRole


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UserServiceConfig(BaseModel):
    """用户服务配置"""
    storage_path: str = Field("./data/users", description="用户数据存储路径")
    enable_cache: bool = Field(True, description="是否启用缓存")
    cache_ttl: int = Field(3600, description="缓存生存时间(秒)")
    auto_save_interval: int = Field(300, description="自动保存间隔(秒)")


class UserService:
    """用户服务类"""
    
    def __init__(self, config: UserServiceConfig):
        """
        初始化用户服务
        
        Args:
            config: 用户服务配置
        """
        self.config = config
        self.users_cache = {}  # 用户缓存，键为用户ID
        self.username_index = {}  # 用户名索引，键为用户名，值为用户ID
        self.email_index = {}  # 邮箱索引，键为邮箱，值为用户ID
        self.cache_time = {}  # 缓存时间，键为用户ID，值为缓存时间
        self.lock = threading.RLock()  # 用于线程安全
        
        # 创建存储目录
        os.makedirs(config.storage_path, exist_ok=True)
        
        # 加载所有用户
        self._load_all_users()
        
        # 启动自动保存任务
        if config.auto_save_interval > 0:
            self._start_auto_save()
        
        logger.info(f"用户服务初始化完成，存储路径: {config.storage_path}")
    
    def create(self, user: User) -> bool:
        """
        创建新用户
        
        Args:
            user: 用户对象
            
        Returns:
            是否创建成功
        """
        with self.lock:
            # 检查用户名和邮箱是否已存在
            if user.username in self.username_index:
                logger.warning(f"用户名已存在: {user.username}")
                return False
            
            if user.email in self.email_index:
                logger.warning(f"邮箱已存在: {user.email}")
                return False
            
            # 保存用户
            success = self._save_user(user)
            if not success:
                return False
            
            # 更新缓存和索引
            self.users_cache[user.id] = user
            self.username_index[user.username] = user.id
            self.email_index[user.email] = user.id
            self.cache_time[user.id] = time.time()
            
            logger.info(f"创建用户成功: {user.id}, 用户名: {user.username}")
            return True
    
    def update(self, user: User) -> bool:
        """
        更新用户信息
        
        Args:
            user: 用户对象
            
        Returns:
            是否更新成功
        """
        with self.lock:
            # 检查用户是否存在
            if user.id not in self.users_cache:
                logger.warning(f"用户ID不存在: {user.id}")
                return False
            
            # 获取旧用户信息
            old_user = self.users_cache[user.id]
            
            # 检查用户名是否已被其他用户使用
            if user.username != old_user.username and user.username in self.username_index:
                if self.username_index[user.username] != user.id:
                    logger.warning(f"用户名已被其他用户使用: {user.username}")
                    return False
            
            # 检查邮箱是否已被其他用户使用
            if user.email != old_user.email and user.email in self.email_index:
                if self.email_index[user.email] != user.id:
                    logger.warning(f"邮箱已被其他用户使用: {user.email}")
                    return False
            
            # 保存用户
            success = self._save_user(user)
            if not success:
                return False
            
            # 更新缓存和索引
            if user.username != old_user.username:
                del self.username_index[old_user.username]
                self.username_index[user.username] = user.id
            
            if user.email != old_user.email:
                del self.email_index[old_user.email]
                self.email_index[user.email] = user.id
            
            self.users_cache[user.id] = user
            self.cache_time[user.id] = time.time()
            
            logger.info(f"更新用户成功: {user.id}")
            return True
    
    def delete(self, user_id: str) -> bool:
        """
        删除用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否删除成功
        """
        with self.lock:
            # 检查用户是否存在
            if user_id not in self.users_cache:
                logger.warning(f"用户ID不存在: {user_id}")
                return False
            
            # 获取用户信息
            user = self.users_cache[user_id]
            
            # 删除用户文件
            user_file = os.path.join(self.config.storage_path, f"{user_id}.json")
            try:
                if os.path.exists(user_file):
                    os.remove(user_file)
            except Exception as e:
                logger.error(f"删除用户文件失败: {str(e)}")
                return False
            
            # 更新缓存和索引
            del self.users_cache[user_id]
            del self.username_index[user.username]
            del self.email_index[user.email]
            if user_id in self.cache_time:
                del self.cache_time[user_id]
            
            logger.info(f"删除用户成功: {user_id}")
            return True
    
    def find_by_id(self, user_id: str) -> Optional[User]:
        """
        根据ID查找用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户对象，如果不存在则返回None
        """
        with self.lock:
            # 检查缓存
            if user_id in self.users_cache:
                # 检查缓存是否过期
                if self.config.enable_cache and time.time() - self.cache_time[user_id] > self.config.cache_ttl:
                    # 重新加载用户
                    self._load_user(user_id)
                return self.users_cache.get(user_id)
            
            # 尝试加载用户
            return self._load_user(user_id)
    
    def find_by_username(self, username: str) -> Optional[User]:
        """
        根据用户名查找用户
        
        Args:
            username: 用户名
            
        Returns:
            用户对象，如果不存在则返回None
        """
        with self.lock:
            # 检查用户名索引
            if username in self.username_index:
                user_id = self.username_index[username]
                return self.find_by_id(user_id)
            return None
    
    def find_by_email(self, email: str) -> Optional[User]:
        """
        根据邮箱查找用户
        
        Args:
            email: 邮箱
            
        Returns:
            用户对象，如果不存在则返回None
        """
        with self.lock:
            # 检查邮箱索引
            if email in self.email_index:
                user_id = self.email_index[email]
                return self.find_by_id(user_id)
            return None
    
    def find_by_role(self, role: UserRole) -> List[User]:
        """
        根据角色查找用户
        
        Args:
            role: 用户角色
            
        Returns:
            用户对象列表
        """
        with self.lock:
            return [user for user in self.users_cache.values() if user.role == role]
    
    def count(self) -> int:
        """
        获取用户总数
        
        Returns:
            用户总数
        """
        with self.lock:
            return len(self.users_cache)
    
    def save_all(self) -> int:
        """
        保存所有用户
        
        Returns:
            保存成功的用户数量
        """
        count = 0
        with self.lock:
            for user_id, user in list(self.users_cache.items()):
                if self._save_user(user):
                    count += 1
        logger.info(f"保存用户完成，成功: {count}, 总数: {len(self.users_cache)}")
        return count
    
    def _save_user(self, user: User) -> bool:
        """
        保存用户到文件
        
        Args:
            user: 用户对象
            
        Returns:
            是否保存成功
        """
        user_file = os.path.join(self.config.storage_path, f"{user.id}.json")
        try:
            with open(user_file, 'w', encoding='utf-8') as f:
                f.write(user.json(ensure_ascii=False, indent=2))
            return True
        except Exception as e:
            logger.error(f"保存用户失败: {user.id}, 错误: {str(e)}")
            return False
    
    def _load_user(self, user_id: str) -> Optional[User]:
        """
        从文件加载用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户对象，如果不存在则返回None
        """
        user_file = os.path.join(self.config.storage_path, f"{user_id}.json")
        if not os.path.exists(user_file):
            return None
        
        try:
            with open(user_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            user = User(**user_data)
            
            # 更新缓存和索引
            self.users_cache[user.id] = user
            self.username_index[user.username] = user.id
            self.email_index[user.email] = user.id
            self.cache_time[user.id] = time.time()
            
            return user
        except Exception as e:
            logger.error(f"加载用户失败: {user_id}, 错误: {str(e)}")
            return None
    
    def _load_all_users(self) -> int:
        """
        加载所有用户
        
        Returns:
            加载的用户数量
        """
        count = 0
        for filename in os.listdir(self.config.storage_path):
            if filename.endswith('.json'):
                user_id = filename[:-5]  # 去掉.json后缀
                user = self._load_user(user_id)
                if user:
                    count += 1
        
        logger.info(f"加载用户完成，成功: {count}")
        return count
    
    def _start_auto_save(self):
        """启动自动保存任务"""
        def auto_save():
            while True:
                time.sleep(self.config.auto_save_interval)
                try:
                    self.save_all()
                except Exception as e:
                    logger.error(f"自动保存失败: {str(e)}")
        
        # 启动自动保存线程
        save_thread = threading.Thread(target=auto_save, daemon=True)
        save_thread.start()
        logger.info(f"自动保存任务已启动，间隔: {self.config.auto_save_interval}秒")


def create_user_service(config: Dict = None) -> UserService:
    """
    创建用户服务实例
    
    Args:
        config: 配置字典
        
    Returns:
        用户服务实例
    """
    if config is None:
        config = {}
    
    # 设置默认存储路径
    if 'storage_path' not in config:
        config['storage_path'] = os.getenv(
            'USER_STORAGE_PATH',
            './data/users'
        )
    
    service_config = UserServiceConfig(**config)
    return UserService(service_config) 