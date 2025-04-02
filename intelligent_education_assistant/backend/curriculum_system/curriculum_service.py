#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
课程体系服务模块

提供课程体系管理、知识点过滤和学习路径规划等功能。
"""

import os
import json
import time
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union

from .knowledge_model import (
    Subject, GradeLevel, KnowledgePoint, KnowledgeRelation, 
    RelationType, Curriculum
)
from .knowledge_graph import KnowledgeGraph, create_knowledge_graph


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CurriculumServiceConfig:
    """课程体系服务配置"""
    
    def __init__(
        self, 
        storage_path: str = "./data/curriculum",
        knowledge_data_path: str = "./data/curriculum/init_data",
        enable_cache: bool = True,
        auto_save_interval: int = 300
    ):
        """
        初始化课程体系服务配置
        
        Args:
            storage_path: 存储路径
            knowledge_data_path: 初始知识数据路径
            enable_cache: 是否启用缓存
            auto_save_interval: 自动保存间隔(秒)
        """
        self.storage_path = storage_path
        self.knowledge_data_path = knowledge_data_path
        self.enable_cache = enable_cache
        self.auto_save_interval = auto_save_interval


class CurriculumService:
    """课程体系服务类"""
    
    def __init__(self, config: CurriculumServiceConfig):
        """
        初始化课程体系服务
        
        Args:
            config: 课程体系服务配置
        """
        self.config = config
        
        # 创建知识图谱
        kg_config = {
            "storage_path": config.storage_path,
            "enable_cache": config.enable_cache,
            "auto_save_interval": config.auto_save_interval
        }
        self.knowledge_graph = create_knowledge_graph(kg_config)
        
        # 创建初始知识数据目录
        os.makedirs(config.knowledge_data_path, exist_ok=True)
        
        # 加载初始知识数据
        self._load_initial_data()
        
        logger.info(f"课程体系服务初始化完成，存储路径: {config.storage_path}")
    
    def add_knowledge_point(self, kp: KnowledgePoint) -> bool:
        """
        添加知识点
        
        Args:
            kp: 知识点对象
            
        Returns:
            是否添加成功
        """
        return self.knowledge_graph.add_knowledge_point(kp)
    
    def update_knowledge_point(self, kp: KnowledgePoint) -> bool:
        """
        更新知识点
        
        Args:
            kp: 知识点对象
            
        Returns:
            是否更新成功
        """
        return self.knowledge_graph.update_knowledge_point(kp)
    
    def delete_knowledge_point(self, kp_id: str) -> bool:
        """
        删除知识点
        
        Args:
            kp_id: 知识点ID
            
        Returns:
            是否删除成功
        """
        return self.knowledge_graph.delete_knowledge_point(kp_id)
    
    def get_knowledge_point(self, kp_id: str) -> Optional[KnowledgePoint]:
        """
        获取知识点
        
        Args:
            kp_id: 知识点ID
            
        Returns:
            知识点对象
        """
        return self.knowledge_graph.get_knowledge_point(kp_id)
    
    def add_knowledge_relation(self, relation: KnowledgeRelation) -> bool:
        """
        添加知识点关系
        
        Args:
            relation: 关系对象
            
        Returns:
            是否添加成功
        """
        return self.knowledge_graph.add_relation(relation)
    
    def delete_knowledge_relation(
        self, 
        source_id: str, 
        target_id: str, 
        relation_type: RelationType
    ) -> bool:
        """
        删除知识点关系
        
        Args:
            source_id: 源知识点ID
            target_id: 目标知识点ID
            relation_type: 关系类型
            
        Returns:
            是否删除成功
        """
        return self.knowledge_graph.delete_relation(source_id, target_id, relation_type)
    
    def get_knowledge_points_by_subject(self, subject: Subject) -> List[KnowledgePoint]:
        """
        根据学科获取知识点
        
        Args:
            subject: 学科
            
        Returns:
            知识点列表
        """
        return self.knowledge_graph.find_knowledge_points_by_subject(subject)
    
    def get_knowledge_points_by_grade(self, grade_level: GradeLevel) -> List[KnowledgePoint]:
        """
        根据年级获取知识点
        
        Args:
            grade_level: 年级
            
        Returns:
            知识点列表
        """
        return self.knowledge_graph.find_knowledge_points_by_grade(grade_level)
    
    def get_knowledge_points_by_subject_and_grade(
        self, 
        subject: Subject, 
        grade_level: GradeLevel
    ) -> List[KnowledgePoint]:
        """
        根据学科和年级获取知识点
        
        Args:
            subject: 学科
            grade_level: 年级
            
        Returns:
            知识点列表
        """
        return self.knowledge_graph.find_knowledge_points_by_subject_and_grade(subject, grade_level)
    
    def get_prerequisite_knowledge_points(self, kp_id: str) -> List[KnowledgePoint]:
        """
        获取前置知识点
        
        Args:
            kp_id: 知识点ID
            
        Returns:
            前置知识点列表
        """
        return self.knowledge_graph.find_prerequisite_knowledge_points(kp_id)
    
    def get_related_knowledge_points(
        self, 
        kp_id: str, 
        relation_type: Optional[RelationType] = None
    ) -> List[KnowledgePoint]:
        """
        获取相关知识点
        
        Args:
            kp_id: 知识点ID
            relation_type: 关系类型
            
        Returns:
            相关知识点列表
        """
        return self.knowledge_graph.find_related_knowledge_points(kp_id, relation_type)
    
    def get_knowledge_children(self, parent_id: str) -> List[KnowledgePoint]:
        """
        获取子知识点
        
        Args:
            parent_id: 父知识点ID
            
        Returns:
            子知识点列表
        """
        return self.knowledge_graph.find_children(parent_id)
    
    def export_knowledge_data(self, subject: Optional[Subject] = None) -> List[Dict]:
        """
        导出知识数据
        
        Args:
            subject: 学科
            
        Returns:
            知识数据
        """
        return self.knowledge_graph.export_knowledge_points(subject)
    
    def import_knowledge_data(self, data: List[Dict]) -> Tuple[int, int]:
        """
        导入知识数据
        
        Args:
            data: 知识数据
            
        Returns:
            (成功数, 失败数)
        """
        return self.knowledge_graph.import_knowledge_points(data)
    
    def plan_learning_path(
        self, 
        target_kp_id: str, 
        known_kp_ids: List[str] = None
    ) -> List[List[str]]:
        """
        规划学习路径
        
        Args:
            target_kp_id: 目标知识点ID
            known_kp_ids: 已掌握的知识点ID列表
            
        Returns:
            学习路径列表，每个路径是知识点ID的列表
        """
        if known_kp_ids is None:
            known_kp_ids = []
        
        # 获取目标知识点
        target_kp = self.knowledge_graph.get_knowledge_point(target_kp_id)
        if not target_kp:
            logger.warning(f"目标知识点不存在: {target_kp_id}")
            return []
        
        # 获取前置知识点
        prereq_kps = []
        visited = set(known_kp_ids)
        queue = [target_kp_id]
        
        while queue:
            current_id = queue.pop(0)
            current_kp = self.knowledge_graph.get_knowledge_point(current_id)
            if not current_kp:
                continue
            
            # 获取直接前置知识点
            direct_prereqs = self.knowledge_graph.find_prerequisite_knowledge_points(current_id)
            
            for prereq in direct_prereqs:
                if prereq.id not in visited:
                    visited.add(prereq.id)
                    queue.append(prereq.id)
                    prereq_kps.append(prereq)
        
        # 按年级和难度排序
        prereq_kps.sort(key=lambda kp: (
            GradeLevel.to_numeric(kp.grade_level),
            kp.difficulty
        ))
        
        # 构建学习路径
        if not prereq_kps:
            # 如果没有前置知识点，直接返回目标知识点
            return [[target_kp_id]]
        
        # 尝试找到从最基础知识点到目标知识点的路径
        paths = []
        for prereq in prereq_kps:
            if not any(prereq.id in p for p in paths):
                for path in self.knowledge_graph.get_knowledge_path(
                    prereq.id, target_kp_id, RelationType.PREREQUISITE
                ):
                    if path not in paths:
                        paths.append(path)
        
        # 如果没有找到路径，构建一个简单路径
        if not paths:
            simple_path = [kp.id for kp in prereq_kps]
            simple_path.append(target_kp_id)
            paths = [simple_path]
        
        return paths
    
    def filter_knowledge_by_criteria(
        self,
        subject: Optional[Subject] = None,
        grade_level: Optional[GradeLevel] = None,
        difficulty_min: float = 0.0,
        difficulty_max: float = 1.0,
        importance_min: float = 0.0,
        keywords: List[str] = None
    ) -> List[KnowledgePoint]:
        """
        根据条件过滤知识点
        
        Args:
            subject: 学科
            grade_level: 年级
            difficulty_min: 最小难度
            difficulty_max: 最大难度
            importance_min: 最小重要性
            keywords: 关键词列表
            
        Returns:
            知识点列表
        """
        # 基于学科和年级初步筛选
        if subject and grade_level:
            kp_list = self.get_knowledge_points_by_subject_and_grade(subject, grade_level)
        elif subject:
            kp_list = self.get_knowledge_points_by_subject(subject)
        elif grade_level:
            kp_list = self.get_knowledge_points_by_grade(grade_level)
        else:
            kp_list = list(self.knowledge_graph.knowledge_points.values())
        
        # 应用其他过滤条件
        result = []
        for kp in kp_list:
            # 难度过滤
            if kp.difficulty < difficulty_min or kp.difficulty > difficulty_max:
                continue
            
            # 重要性过滤
            if kp.importance < importance_min:
                continue
            
            # 关键词过滤
            if keywords:
                if not any(keyword.lower() in kp.name.lower() or 
                          any(keyword.lower() in kw.lower() for kw in kp.keywords) 
                          for keyword in keywords):
                    continue
            
            result.append(kp)
        
        return result
    
    def get_curriculum_structure(
        self, 
        subject: Subject, 
        grade_level: GradeLevel
    ) -> Dict[str, Any]:
        """
        获取课程结构
        
        Args:
            subject: 学科
            grade_level: 年级
            
        Returns:
            课程结构数据
        """
        # 获取学科和年级的知识点
        kp_list = self.get_knowledge_points_by_subject_and_grade(subject, grade_level)
        
        # 找出顶级知识点（没有父节点的）
        root_kps = [kp for kp in kp_list if not kp.parent_id]
        
        # 递归构建知识树
        def build_tree(kp):
            children = self.get_knowledge_children(kp.id)
            return {
                "id": kp.id,
                "name": kp.name,
                "difficulty": kp.difficulty,
                "importance": kp.importance,
                "description": kp.description,
                "children": [build_tree(child) for child in children]
            }
        
        # 构建课程树
        curriculum_tree = [build_tree(kp) for kp in root_kps]
        
        return {
            "subject": subject,
            "grade_level": grade_level,
            "knowledge_tree": curriculum_tree,
            "total_count": len(kp_list),
            "root_count": len(root_kps)
        }
    
    def save_initial_data(self, subject: Subject, data: List[Dict]) -> bool:
        """
        保存初始知识数据
        
        Args:
            subject: 学科
            data: 知识数据
            
        Returns:
            是否保存成功
        """
        file_path = os.path.join(self.config.knowledge_data_path, f"{subject}.json")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存初始知识数据失败: {subject}, 错误: {str(e)}")
            return False
    
    def _load_initial_data(self) -> None:
        """加载初始知识数据"""
        data_dir = self.config.knowledge_data_path
        if not os.path.exists(data_dir):
            logger.info(f"初始知识数据目录不存在: {data_dir}")
            return
        
        # 遍历数据目录下的JSON文件
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 导入数据
                    success, fail = self.import_knowledge_data(data)
                    logger.info(f"加载初始知识数据: {filename}, 成功: {success}, 失败: {fail}")
                except Exception as e:
                    logger.error(f"加载初始知识数据失败: {filename}, 错误: {str(e)}")


def create_curriculum_service(config: Dict = None) -> CurriculumService:
    """
    创建课程体系服务实例
    
    Args:
        config: 配置字典
        
    Returns:
        课程体系服务实例
    """
    if config is None:
        config = {}
    
    # 设置默认存储路径
    if 'storage_path' not in config:
        config['storage_path'] = os.getenv(
            'CURRICULUM_STORAGE_PATH',
            './data/curriculum'
        )
    
    service_config = CurriculumServiceConfig(
        storage_path=config.get('storage_path', './data/curriculum'),
        knowledge_data_path=config.get('knowledge_data_path', './data/curriculum/init_data'),
        enable_cache=config.get('enable_cache', True),
        auto_save_interval=config.get('auto_save_interval', 300)
    )
    
    return CurriculumService(service_config) 