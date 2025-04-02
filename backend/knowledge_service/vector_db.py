#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
向量数据库集成模块

该模块实现了与FAISS向量数据库的集成，用于存储和检索知识项的向量表示。
支持向量的添加、检索和删除等基本操作。
"""

import os
import time
import logging
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

import faiss
from pydantic import BaseModel, Field

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorDBConfig(BaseModel):
    """向量数据库配置"""
    dimension: int = Field(768, description="向量维度")
    index_type: str = Field("IVF100,Flat", description="FAISS索引类型")
    nlist: int = Field(100, description="IVF聚类中心数量")
    metric_type: str = Field("L2", description="距离度量类型")
    use_gpu: bool = Field(False, description="是否使用GPU")
    db_path: str = Field("./data/vector_db", description="数据库存储路径")


class FAISSVectorDB:
    """FAISS向量数据库接口实现"""
    
    def __init__(self, config: VectorDBConfig):
        """
        初始化FAISS向量数据库
        
        Args:
            config: 向量数据库配置
        """
        self.config = config
        self.dimension = config.dimension
        self.index = None
        self.id_map = {}  # 映射内部ID到外部ID
        self.reverse_id_map = {}  # 映射外部ID到内部ID
        self.next_id = 0
        
        # 创建数据库目录
        os.makedirs(config.db_path, exist_ok=True)
        
        # 初始化索引
        self._init_index()
        
        logger.info(f"初始化FAISS向量数据库 - 维度: {self.dimension}, 索引类型: {config.index_type}")
    
    def _init_index(self):
        """初始化FAISS索引"""
        try:
            # 尝试加载现有索引
            if self._load_index():
                logger.info("成功加载现有索引")
                return
        except Exception as e:
            logger.warning(f"加载现有索引失败: {str(e)}，将创建新索引")
        
        # 创建新索引
        if self.config.index_type == "Flat":
            # 简单的暴力搜索索引
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.config.index_type.startswith("IVF"):
            # 创建IVF索引
            parts = self.config.index_type.split(",")
            nlist = int(parts[0].replace("IVF", ""))
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            # 需要训练
            self.trained = False
        else:
            # 默认使用Flat索引
            logger.warning(f"不支持的索引类型: {self.config.index_type}，使用默认Flat索引")
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # 使用GPU加速
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            logger.info("启用GPU加速")
    
    def _save_index(self):
        """保存索引到文件"""
        if self.index is None:
            logger.warning("索引未初始化，无法保存")
            return False
        
        try:
            # 如果是GPU索引，转回CPU
            if self.config.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            else:
                cpu_index = self.index
            
            # 保存索引
            index_path = os.path.join(self.config.db_path, "faiss_index.bin")
            faiss.write_index(cpu_index, index_path)
            
            # 保存ID映射
            id_map_path = os.path.join(self.config.db_path, "id_map.pkl")
            with open(id_map_path, 'wb') as f:
                pickle.dump({
                    'id_map': self.id_map,
                    'reverse_id_map': self.reverse_id_map,
                    'next_id': self.next_id
                }, f)
            
            logger.info(f"索引已保存到: {index_path}")
            return True
        except Exception as e:
            logger.error(f"保存索引失败: {str(e)}", exc_info=True)
            return False
    
    def _load_index(self):
        """从文件加载索引"""
        index_path = os.path.join(self.config.db_path, "faiss_index.bin")
        id_map_path = os.path.join(self.config.db_path, "id_map.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(id_map_path):
            logger.warning("索引文件不存在，无法加载")
            return False
        
        try:
            # 加载索引
            self.index = faiss.read_index(index_path)
            
            # 如果使用GPU，转到GPU
            if self.config.use_gpu and faiss.get_num_gpus() > 0:
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            
            # 加载ID映射
            with open(id_map_path, 'rb') as f:
                data = pickle.load(f)
                self.id_map = data['id_map']
                self.reverse_id_map = data['reverse_id_map']
                self.next_id = data['next_id']
            
            logger.info(f"成功从{index_path}加载索引，包含{self.next_id}个向量")
            return True
        except Exception as e:
            logger.error(f"加载索引失败: {str(e)}", exc_info=True)
            return False
    
    def add(self, id: str, vector: np.ndarray) -> bool:
        """
        添加向量到索引
        
        Args:
            id: 向量ID
            vector: 向量数据
            
        Returns:
            是否添加成功
        """
        if self.index is None:
            logger.error("索引未初始化，无法添加向量")
            return False
        
        if id in self.reverse_id_map:
            logger.warning(f"ID '{id}'已存在，先删除旧向量")
            self.delete(id)
        
        # 确保向量是正确的维度和格式
        if len(vector) != self.dimension:
            logger.error(f"向量维度不匹配: 预期{self.dimension}，实际{len(vector)}")
            return False
        
        vector = np.array(vector).astype('float32').reshape(1, -1)
        
        # 添加向量到索引
        try:
            # 如果是IVF索引且未训练，需要先训练
            if hasattr(self, 'trained') and not self.trained and self.next_id >= 100:
                logger.info("训练IVF索引...")
                self.index.train(np.array(list(self.id_map.values())).astype('float32'))
                self.trained = True
            
            # 添加向量
            self.index.add(vector)
            
            # 更新ID映射
            self.id_map[self.next_id] = id
            self.reverse_id_map[id] = self.next_id
            self.next_id += 1
            
            # 定期保存索引
            if self.next_id % 100 == 0:
                self._save_index()
            
            return True
        except Exception as e:
            logger.error(f"添加向量失败: {str(e)}", exc_info=True)
            return False
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        搜索最相似的向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回的最大结果数量
            
        Returns:
            (ID, 相似度分数)的列表，按相似度降序排列
        """
        if self.index is None:
            logger.error("索引未初始化，无法搜索")
            return []
        
        if len(query_vector) != self.dimension:
            logger.error(f"查询向量维度不匹配: 预期{self.dimension}，实际{len(query_vector)}")
            return []
        
        # 确保向量是正确的维度和格式
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        
        try:
            # 如果索引为空，返回空结果
            if self.next_id == 0:
                return []
            
            # 执行搜索
            distances, indices = self.index.search(query_vector, min(top_k, self.next_id))
            
            # 将结果映射回原始ID
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx in self.id_map:  # -1表示无效结果
                    # 转换距离为相似度分数 (1.0 - 归一化距离)
                    distance = distances[0][i]
                    max_distance = 100.0  # 最大距离阈值
                    similarity = 1.0 - min(distance / max_distance, 1.0)
                    results.append((self.id_map[idx], float(similarity)))
            
            return results
        except Exception as e:
            logger.error(f"搜索向量失败: {str(e)}", exc_info=True)
            return []
    
    def delete(self, id: str) -> bool:
        """
        删除向量
        
        Args:
            id: 要删除的向量ID
            
        Returns:
            是否删除成功
        """
        if self.index is None:
            logger.error("索引未初始化，无法删除向量")
            return False
        
        if id not in self.reverse_id_map:
            logger.warning(f"ID '{id}'不存在，无法删除")
            return False
        
        try:
            # 目前FAISS不支持直接删除，需要重建索引
            # 这里采用标记删除的方式，从映射中删除，但索引中依然存在
            internal_id = self.reverse_id_map[id]
            del self.id_map[internal_id]
            del self.reverse_id_map[id]
            
            logger.info(f"标记删除向量: {id}")
            
            # 定期重建索引
            if len(self.reverse_id_map) < self.next_id * 0.7:
                logger.info("索引存在大量删除向量，计划重建索引")
                # TODO: 在后台任务中重建索引
            
            return True
        except Exception as e:
            logger.error(f"删除向量失败: {str(e)}", exc_info=True)
            return False
    
    def clear(self) -> bool:
        """
        清空索引
        
        Returns:
            是否清空成功
        """
        try:
            # 重新初始化索引
            self._init_index()
            self.id_map = {}
            self.reverse_id_map = {}
            self.next_id = 0
            
            # 删除索引文件
            index_path = os.path.join(self.config.db_path, "faiss_index.bin")
            id_map_path = os.path.join(self.config.db_path, "id_map.pkl")
            
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(id_map_path):
                os.remove(id_map_path)
            
            logger.info("索引已清空")
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
        if self.index is None:
            return {"status": "未初始化"}
        
        return {
            "total_vectors": self.next_id,
            "active_vectors": len(self.reverse_id_map),
            "dimension": self.dimension,
            "index_type": self.config.index_type,
            "use_gpu": self.config.use_gpu
        }


def create_vector_db(config: Dict = None) -> FAISSVectorDB:
    """
    创建向量数据库实例
    
    Args:
        config: 配置字典
        
    Returns:
        向量数据库实例
    """
    if config is None:
        config = {}
    
    db_config = VectorDBConfig(**config)
    return FAISSVectorDB(db_config) 