#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
知识图谱模块

提供知识图谱构建、查询和分析功能，用于管理知识点之间的关系。
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict

from .knowledge_model import (
    Subject, GradeLevel, KnowledgePoint, KnowledgeRelation, RelationType
)


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeGraphConfig:
    """知识图谱配置"""
    
    def __init__(
        self, 
        storage_path: str = "./data/curriculum",
        enable_cache: bool = True,
        auto_save_interval: int = 300
    ):
        """
        初始化知识图谱配置
        
        Args:
            storage_path: 存储路径
            enable_cache: 是否启用缓存
            auto_save_interval: 自动保存间隔(秒)
        """
        self.storage_path = storage_path
        self.enable_cache = enable_cache
        self.auto_save_interval = auto_save_interval


class KnowledgeGraph:
    """知识图谱类"""
    
    def __init__(self, config: KnowledgeGraphConfig):
        """
        初始化知识图谱
        
        Args:
            config: 知识图谱配置
        """
        self.config = config
        self.lock = threading.RLock()  # 用于线程安全
        
        # 数据存储
        self.knowledge_points: Dict[str, KnowledgePoint] = {}  # 知识点字典，键为ID
        self.relations: Dict[str, KnowledgeRelation] = {}  # 关系字典，键为关系键
        
        # 索引
        self.subject_index: Dict[Subject, Set[str]] = defaultdict(set)  # 学科索引，值为知识点ID集合
        self.grade_index: Dict[GradeLevel, Set[str]] = defaultdict(set)  # 年级索引，值为知识点ID集合
        self.parent_index: Dict[str, Set[str]] = defaultdict(set)  # 父节点索引，键为父ID，值为子ID集合
        
        # 图结构索引
        self.outgoing_edges: Dict[str, Dict[RelationType, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.incoming_edges: Dict[str, Dict[RelationType, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        # 创建存储目录
        os.makedirs(config.storage_path, exist_ok=True)
        os.makedirs(os.path.join(config.storage_path, "knowledge_points"), exist_ok=True)
        os.makedirs(os.path.join(config.storage_path, "relations"), exist_ok=True)
        
        # 加载数据
        self._load_all_data()
        
        # 启动自动保存任务
        if config.auto_save_interval > 0:
            self._start_auto_save()
        
        logger.info(f"知识图谱初始化完成，存储路径: {config.storage_path}")
    
    def add_knowledge_point(self, kp: KnowledgePoint) -> bool:
        """
        添加知识点
        
        Args:
            kp: 知识点对象
            
        Returns:
            是否添加成功
        """
        with self.lock:
            # 检查是否已存在
            if kp.id in self.knowledge_points:
                logger.warning(f"知识点已存在: {kp.id}")
                return False
            
            # 保存知识点
            success = self._save_knowledge_point(kp)
            if not success:
                return False
            
            # 更新内存和索引
            self.knowledge_points[kp.id] = kp
            self.subject_index[kp.subject].add(kp.id)
            self.grade_index[kp.grade_level].add(kp.id)
            
            # 更新父节点索引
            if kp.parent_id:
                self.parent_index[kp.parent_id].add(kp.id)
            
            logger.info(f"添加知识点成功: {kp.id}, 名称: {kp.name}")
            return True
    
    def update_knowledge_point(self, kp: KnowledgePoint) -> bool:
        """
        更新知识点
        
        Args:
            kp: 知识点对象
            
        Returns:
            是否更新成功
        """
        with self.lock:
            # 检查是否存在
            if kp.id not in self.knowledge_points:
                logger.warning(f"知识点不存在: {kp.id}")
                return False
            
            old_kp = self.knowledge_points[kp.id]
            
            # 保存知识点
            success = self._save_knowledge_point(kp)
            if not success:
                return False
            
            # 更新内存和索引
            # 如果学科变更，更新学科索引
            if old_kp.subject != kp.subject:
                self.subject_index[old_kp.subject].remove(kp.id)
                self.subject_index[kp.subject].add(kp.id)
            
            # 如果年级变更，更新年级索引
            if old_kp.grade_level != kp.grade_level:
                self.grade_index[old_kp.grade_level].remove(kp.id)
                self.grade_index[kp.grade_level].add(kp.id)
            
            # 如果父节点变更，更新父节点索引
            if old_kp.parent_id != kp.parent_id:
                if old_kp.parent_id:
                    self.parent_index[old_kp.parent_id].remove(kp.id)
                if kp.parent_id:
                    self.parent_index[kp.parent_id].add(kp.id)
            
            # 更新知识点
            self.knowledge_points[kp.id] = kp
            
            logger.info(f"更新知识点成功: {kp.id}")
            return True
    
    def delete_knowledge_point(self, kp_id: str) -> bool:
        """
        删除知识点
        
        Args:
            kp_id: 知识点ID
            
        Returns:
            是否删除成功
        """
        with self.lock:
            # 检查是否存在
            if kp_id not in self.knowledge_points:
                logger.warning(f"知识点不存在: {kp_id}")
                return False
            
            kp = self.knowledge_points[kp_id]
            
            # 删除关系
            related_relations = self.find_relations_by_knowledge_point(kp_id)
            for relation in related_relations:
                self.delete_relation(relation.source_id, relation.target_id, relation.relation_type)
            
            # 删除文件
            file_path = os.path.join(self.config.storage_path, "knowledge_points", f"{kp_id}.json")
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"删除知识点文件失败: {str(e)}")
                return False
            
            # 更新内存和索引
            del self.knowledge_points[kp_id]
            self.subject_index[kp.subject].remove(kp_id)
            self.grade_index[kp.grade_level].remove(kp_id)
            
            # 更新父节点索引
            if kp.parent_id and kp_id in self.parent_index[kp.parent_id]:
                self.parent_index[kp.parent_id].remove(kp_id)
            
            # 删除所有子节点的父节点引用
            for child_id in list(self.parent_index.get(kp_id, [])):
                child = self.knowledge_points.get(child_id)
                if child:
                    child.parent_id = None
                    self.update_knowledge_point(child)
            
            if kp_id in self.parent_index:
                del self.parent_index[kp_id]
            
            # 删除图结构索引
            if kp_id in self.outgoing_edges:
                del self.outgoing_edges[kp_id]
            if kp_id in self.incoming_edges:
                del self.incoming_edges[kp_id]
            
            logger.info(f"删除知识点成功: {kp_id}")
            return True
    
    def add_relation(self, relation: KnowledgeRelation) -> bool:
        """
        添加知识点关系
        
        Args:
            relation: 关系对象
            
        Returns:
            是否添加成功
        """
        with self.lock:
            # 检查知识点是否存在
            if relation.source_id not in self.knowledge_points:
                logger.warning(f"源知识点不存在: {relation.source_id}")
                return False
            
            if relation.target_id not in self.knowledge_points:
                logger.warning(f"目标知识点不存在: {relation.target_id}")
                return False
            
            # 获取关系键
            relation_key = relation.get_key()
            
            # 检查关系是否已存在
            if relation_key in self.relations:
                logger.warning(f"关系已存在: {relation_key}")
                return False
            
            # 保存关系
            success = self._save_relation(relation)
            if not success:
                return False
            
            # 更新内存和索引
            self.relations[relation_key] = relation
            
            # 更新图结构索引
            self.outgoing_edges[relation.source_id][relation.relation_type].append(relation.target_id)
            self.incoming_edges[relation.target_id][relation.relation_type].append(relation.source_id)
            
            logger.info(f"添加关系成功: {relation_key}")
            return True
    
    def update_relation(self, relation: KnowledgeRelation) -> bool:
        """
        更新知识点关系
        
        Args:
            relation: 关系对象
            
        Returns:
            是否更新成功
        """
        with self.lock:
            # 获取关系键
            relation_key = relation.get_key()
            
            # 检查关系是否存在
            if relation_key not in self.relations:
                logger.warning(f"关系不存在: {relation_key}")
                return False
            
            # 保存关系
            success = self._save_relation(relation)
            if not success:
                return False
            
            # 更新内存
            self.relations[relation_key] = relation
            
            logger.info(f"更新关系成功: {relation_key}")
            return True
    
    def delete_relation(self, source_id: str, target_id: str, relation_type: RelationType) -> bool:
        """
        删除知识点关系
        
        Args:
            source_id: 源知识点ID
            target_id: 目标知识点ID
            relation_type: 关系类型
            
        Returns:
            是否删除成功
        """
        with self.lock:
            # 创建临时关系对象以获取键
            temp_relation = KnowledgeRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type
            )
            relation_key = temp_relation.get_key()
            
            # 检查关系是否存在
            if relation_key not in self.relations:
                logger.warning(f"关系不存在: {relation_key}")
                return False
            
            # 删除文件
            file_path = os.path.join(self.config.storage_path, "relations", f"{relation_key}.json")
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"删除关系文件失败: {str(e)}")
                return False
            
            # 更新内存和索引
            relation = self.relations[relation_key]
            del self.relations[relation_key]
            
            # 更新图结构索引
            if relation.target_id in self.outgoing_edges[relation.source_id][relation.relation_type]:
                self.outgoing_edges[relation.source_id][relation.relation_type].remove(relation.target_id)
            
            if relation.source_id in self.incoming_edges[relation.target_id][relation.relation_type]:
                self.incoming_edges[relation.target_id][relation.relation_type].remove(relation.source_id)
            
            logger.info(f"删除关系成功: {relation_key}")
            return True
    
    def get_knowledge_point(self, kp_id: str) -> Optional[KnowledgePoint]:
        """
        获取知识点
        
        Args:
            kp_id: 知识点ID
            
        Returns:
            知识点对象，如果不存在则返回None
        """
        return self.knowledge_points.get(kp_id)
    
    def get_relation(self, source_id: str, target_id: str, relation_type: RelationType) -> Optional[KnowledgeRelation]:
        """
        获取关系
        
        Args:
            source_id: 源知识点ID
            target_id: 目标知识点ID
            relation_type: 关系类型
            
        Returns:
            关系对象，如果不存在则返回None
        """
        # 创建临时关系对象以获取键
        temp_relation = KnowledgeRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type
        )
        relation_key = temp_relation.get_key()
        
        return self.relations.get(relation_key)
    
    def find_knowledge_points_by_subject(self, subject: Subject) -> List[KnowledgePoint]:
        """
        根据学科查找知识点
        
        Args:
            subject: 学科
            
        Returns:
            知识点列表
        """
        kp_ids = self.subject_index.get(subject, set())
        return [self.knowledge_points[kp_id] for kp_id in kp_ids if kp_id in self.knowledge_points]
    
    def find_knowledge_points_by_grade(self, grade_level: GradeLevel) -> List[KnowledgePoint]:
        """
        根据年级查找知识点
        
        Args:
            grade_level: 年级
            
        Returns:
            知识点列表
        """
        kp_ids = self.grade_index.get(grade_level, set())
        return [self.knowledge_points[kp_id] for kp_id in kp_ids if kp_id in self.knowledge_points]
    
    def find_knowledge_points_by_subject_and_grade(
        self, 
        subject: Subject, 
        grade_level: GradeLevel
    ) -> List[KnowledgePoint]:
        """
        根据学科和年级查找知识点
        
        Args:
            subject: 学科
            grade_level: 年级
            
        Returns:
            知识点列表
        """
        subject_kp_ids = self.subject_index.get(subject, set())
        grade_kp_ids = self.grade_index.get(grade_level, set())
        common_ids = subject_kp_ids.intersection(grade_kp_ids)
        
        return [self.knowledge_points[kp_id] for kp_id in common_ids if kp_id in self.knowledge_points]
    
    def find_children(self, parent_id: str) -> List[KnowledgePoint]:
        """
        查找子知识点
        
        Args:
            parent_id: 父知识点ID
            
        Returns:
            子知识点列表
        """
        child_ids = self.parent_index.get(parent_id, set())
        return [self.knowledge_points[child_id] for child_id in child_ids if child_id in self.knowledge_points]
    
    def find_related_knowledge_points(
        self, 
        kp_id: str, 
        relation_type: Optional[RelationType] = None
    ) -> List[KnowledgePoint]:
        """
        查找相关知识点
        
        Args:
            kp_id: 知识点ID
            relation_type: 关系类型，如果为None则返回所有关系类型
            
        Returns:
            相关知识点列表
        """
        if kp_id not in self.outgoing_edges:
            return []
        
        related_ids = set()
        
        if relation_type:
            related_ids.update(self.outgoing_edges[kp_id].get(relation_type, []))
        else:
            for rel_type in self.outgoing_edges[kp_id]:
                related_ids.update(self.outgoing_edges[kp_id][rel_type])
        
        return [self.knowledge_points[related_id] for related_id in related_ids if related_id in self.knowledge_points]
    
    def find_prerequisite_knowledge_points(self, kp_id: str) -> List[KnowledgePoint]:
        """
        查找前置知识点
        
        Args:
            kp_id: 知识点ID
            
        Returns:
            前置知识点列表
        """
        if kp_id not in self.incoming_edges:
            return []
        
        prerequisite_ids = self.incoming_edges[kp_id].get(RelationType.PREREQUISITE, [])
        return [self.knowledge_points[pid] for pid in prerequisite_ids if pid in self.knowledge_points]
    
    def find_relations_by_knowledge_point(self, kp_id: str) -> List[KnowledgeRelation]:
        """
        查找与知识点相关的所有关系
        
        Args:
            kp_id: 知识点ID
            
        Returns:
            关系列表
        """
        result = []
        
        for relation in self.relations.values():
            if relation.source_id == kp_id or relation.target_id == kp_id:
                result.append(relation)
        
        return result
    
    def get_knowledge_path(
        self, 
        start_id: str, 
        end_id: str, 
        relation_type: Optional[RelationType] = None,
        max_depth: int = 5
    ) -> List[List[str]]:
        """
        获取两个知识点之间的路径
        
        Args:
            start_id: 起始知识点ID
            end_id: 目标知识点ID
            relation_type: 关系类型，如果为None则考虑所有关系类型
            max_depth: 最大深度
            
        Returns:
            路径列表，每个路径是知识点ID的列表
        """
        if start_id not in self.knowledge_points or end_id not in self.knowledge_points:
            return []
        
        visited = set()
        path = [start_id]
        all_paths = []
        
        def dfs(current_id, target_id, depth):
            if depth > max_depth:
                return
            
            if current_id == target_id:
                all_paths.append(path.copy())
                return
            
            visited.add(current_id)
            
            if current_id in self.outgoing_edges:
                for rel_type, targets in self.outgoing_edges[current_id].items():
                    if relation_type and rel_type != relation_type:
                        continue
                    
                    for next_id in targets:
                        if next_id not in visited:
                            path.append(next_id)
                            dfs(next_id, target_id, depth + 1)
                            path.pop()
            
            visited.remove(current_id)
        
        dfs(start_id, end_id, 1)
        return all_paths
    
    def export_knowledge_points(self, subject: Optional[Subject] = None) -> List[Dict]:
        """
        导出知识点数据
        
        Args:
            subject: 学科，如果为None则导出所有学科
            
        Returns:
            知识点数据列表
        """
        result = []
        
        if subject:
            kp_list = self.find_knowledge_points_by_subject(subject)
        else:
            kp_list = list(self.knowledge_points.values())
        
        for kp in kp_list:
            kp_data = kp.to_dict()
            kp_data["children"] = [child.id for child in self.find_children(kp.id)]
            kp_data["relations"] = {}
            
            # 添加关系数据
            if kp.id in self.outgoing_edges:
                for rel_type, targets in self.outgoing_edges[kp.id].items():
                    kp_data["relations"][rel_type] = targets
            
            result.append(kp_data)
        
        return result
    
    def import_knowledge_points(self, data_list: List[Dict]) -> Tuple[int, int]:
        """
        导入知识点数据
        
        Args:
            data_list: 知识点数据列表
            
        Returns:
            (成功数, 失败数)
        """
        success_count = 0
        fail_count = 0
        
        with self.lock:
            # 第一遍：导入所有知识点
            for kp_data in data_list:
                try:
                    # 创建知识点对象
                    kp = KnowledgePoint(
                        id=kp_data["id"],
                        name=kp_data["name"],
                        subject=kp_data["subject"],
                        grade_level=kp_data["grade_level"],
                        description=kp_data.get("description", ""),
                        keywords=kp_data.get("keywords", []),
                        difficulty=kp_data.get("difficulty", 0.5),
                        importance=kp_data.get("importance", 0.5),
                        parent_id=kp_data.get("parent_id"),
                        created_at=kp_data.get("created_at", time.time()),
                        updated_at=kp_data.get("updated_at", time.time()),
                        metadata=kp_data.get("metadata", {})
                    )
                    
                    # 添加或更新知识点
                    if kp.id in self.knowledge_points:
                        if self.update_knowledge_point(kp):
                            success_count += 1
                        else:
                            fail_count += 1
                    else:
                        if self.add_knowledge_point(kp):
                            success_count += 1
                        else:
                            fail_count += 1
                except Exception as e:
                    logger.error(f"导入知识点失败: {str(e)}")
                    fail_count += 1
            
            # 第二遍：处理关系
            for kp_data in data_list:
                if "relations" in kp_data and isinstance(kp_data["relations"], dict):
                    source_id = kp_data["id"]
                    
                    for rel_type_str, target_ids in kp_data["relations"].items():
                        try:
                            # 解析关系类型
                            rel_type = RelationType(rel_type_str)
                            
                            # 添加关系
                            for target_id in target_ids:
                                if target_id in self.knowledge_points:
                                    relation = KnowledgeRelation(
                                        source_id=source_id,
                                        target_id=target_id,
                                        relation_type=rel_type
                                    )
                                    
                                    # 检查关系是否已存在
                                    relation_key = relation.get_key()
                                    if relation_key not in self.relations:
                                        self.add_relation(relation)
                        except Exception as e:
                            logger.error(f"处理关系失败: {str(e)}")
        
        return success_count, fail_count
    
    def save_all(self) -> Tuple[int, int]:
        """
        保存所有数据
        
        Returns:
            (知识点成功数, 关系成功数)
        """
        kp_success = 0
        relation_success = 0
        
        with self.lock:
            # 保存知识点
            for kp in self.knowledge_points.values():
                if self._save_knowledge_point(kp):
                    kp_success += 1
            
            # 保存关系
            for relation in self.relations.values():
                if self._save_relation(relation):
                    relation_success += 1
        
        logger.info(f"保存数据完成，知识点: {kp_success}/{len(self.knowledge_points)}, 关系: {relation_success}/{len(self.relations)}")
        return kp_success, relation_success
    
    def _save_knowledge_point(self, kp: KnowledgePoint) -> bool:
        """
        保存知识点到文件
        
        Args:
            kp: 知识点对象
            
        Returns:
            是否保存成功
        """
        file_path = os.path.join(self.config.storage_path, "knowledge_points", f"{kp.id}.json")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(kp.json(ensure_ascii=False, indent=2))
            return True
        except Exception as e:
            logger.error(f"保存知识点失败: {kp.id}, 错误: {str(e)}")
            return False
    
    def _save_relation(self, relation: KnowledgeRelation) -> bool:
        """
        保存关系到文件
        
        Args:
            relation: 关系对象
            
        Returns:
            是否保存成功
        """
        relation_key = relation.get_key()
        file_path = os.path.join(self.config.storage_path, "relations", f"{relation_key}.json")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(relation.json(ensure_ascii=False, indent=2))
            return True
        except Exception as e:
            logger.error(f"保存关系失败: {relation_key}, 错误: {str(e)}")
            return False
    
    def _load_knowledge_point(self, file_path: str) -> Optional[KnowledgePoint]:
        """
        从文件加载知识点
        
        Args:
            file_path: 文件路径
            
        Returns:
            知识点对象，如果加载失败则返回None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            kp = KnowledgePoint(**data)
            
            # 更新内存和索引
            self.knowledge_points[kp.id] = kp
            self.subject_index[kp.subject].add(kp.id)
            self.grade_index[kp.grade_level].add(kp.id)
            
            # 更新父节点索引
            if kp.parent_id:
                self.parent_index[kp.parent_id].add(kp.id)
            
            return kp
        except Exception as e:
            logger.error(f"加载知识点失败: {file_path}, 错误: {str(e)}")
            return None
    
    def _load_relation(self, file_path: str) -> Optional[KnowledgeRelation]:
        """
        从文件加载关系
        
        Args:
            file_path: 文件路径
            
        Returns:
            关系对象，如果加载失败则返回None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            relation = KnowledgeRelation(**data)
            relation_key = relation.get_key()
            
            # 检查知识点是否存在
            if relation.source_id not in self.knowledge_points:
                logger.warning(f"源知识点不存在: {relation.source_id}")
                return None
            
            if relation.target_id not in self.knowledge_points:
                logger.warning(f"目标知识点不存在: {relation.target_id}")
                return None
            
            # 更新内存和索引
            self.relations[relation_key] = relation
            
            # 更新图结构索引
            self.outgoing_edges[relation.source_id][relation.relation_type].append(relation.target_id)
            self.incoming_edges[relation.target_id][relation.relation_type].append(relation.source_id)
            
            return relation
        except Exception as e:
            logger.error(f"加载关系失败: {file_path}, 错误: {str(e)}")
            return None
    
    def _load_all_data(self) -> Tuple[int, int]:
        """
        加载所有数据
        
        Returns:
            (知识点数, 关系数)
        """
        # 加载知识点
        kp_count = 0
        kp_dir = os.path.join(self.config.storage_path, "knowledge_points")
        if os.path.exists(kp_dir):
            for filename in os.listdir(kp_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(kp_dir, filename)
                    if self._load_knowledge_point(file_path):
                        kp_count += 1
        
        # 加载关系
        relation_count = 0
        relation_dir = os.path.join(self.config.storage_path, "relations")
        if os.path.exists(relation_dir):
            for filename in os.listdir(relation_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(relation_dir, filename)
                    if self._load_relation(file_path):
                        relation_count += 1
        
        logger.info(f"加载数据完成，知识点: {kp_count}, 关系: {relation_count}")
        return kp_count, relation_count
    
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


def create_knowledge_graph(config: Dict = None) -> KnowledgeGraph:
    """
    创建知识图谱实例
    
    Args:
        config: 配置字典
        
    Returns:
        知识图谱实例
    """
    if config is None:
        config = {}
    
    # 设置默认存储路径
    if 'storage_path' not in config:
        config['storage_path'] = os.getenv(
            'CURRICULUM_STORAGE_PATH',
            './data/curriculum'
        )
    
    graph_config = KnowledgeGraphConfig(
        storage_path=config.get('storage_path', './data/curriculum'),
        enable_cache=config.get('enable_cache', True),
        auto_save_interval=config.get('auto_save_interval', 300)
    )
    
    return KnowledgeGraph(graph_config) 