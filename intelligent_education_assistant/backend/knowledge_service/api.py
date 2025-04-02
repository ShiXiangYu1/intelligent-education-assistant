from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
import time
import os
from log import logger
from knowledge_service.retriever import KnowledgeRetriever

app = APIRouter()

class SearchRequest(BaseModel):
    """知识检索请求模型"""
    query: str = Field(..., description="检索查询")
    grade_level: Optional[int] = Field(None, description="年级(1-12)")
    subject: Optional[str] = Field(None, description="学科")
    top_k: int = Field(10, description="返回结果数量")
    min_score: float = Field(0.6, description="最小匹配分数")
    apply_curriculum_filter: bool = Field(True, description="是否应用课标过滤")

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
            
            # 内容过滤器配置
            content_filter_config = None
            if os.getenv("ENABLE_CURRICULUM_FILTER", "true").lower() == "true":
                content_filter_config = {
                    "strictness_level": float(os.getenv("CURRICULUM_FILTER_STRICTNESS", "0.7")),
                    "enable_keyword_matching": os.getenv("CURRICULUM_FILTER_KEYWORD_MATCHING", "true").lower() == "true",
                    "enable_concept_matching": os.getenv("CURRICULUM_FILTER_CONCEPT_MATCHING", "true").lower() == "true",
                    "min_matching_keywords": int(os.getenv("CURRICULUM_FILTER_MIN_KEYWORDS", "2")),
                    "min_matching_ratio": float(os.getenv("CURRICULUM_FILTER_MIN_RATIO", "0.3")),
                    "cache_size": int(os.getenv("CURRICULUM_FILTER_CACHE_SIZE", "1000")),
                    "curriculum_service_config": {
                        "storage_path": os.getenv("CURRICULUM_STORAGE_PATH", "./data/curriculum")
                    }
                }
                logger.info("已启用课标内容过滤")
            else:
                logger.info("课标内容过滤已禁用")
            
            # 检索器配置
            vector_weight = float(os.getenv("RETRIEVER_VECTOR_WEIGHT", "0.6"))
            keyword_weight = float(os.getenv("RETRIEVER_KEYWORD_WEIGHT", "0.4"))
            enable_grade_filter = os.getenv("RETRIEVER_ENABLE_GRADE_FILTER", "true").lower() == "true"
            enable_mock_data = os.getenv("RETRIEVER_ENABLE_MOCK_DATA", "true").lower() == "true"
            
            get_retriever.instance = create_retriever(
                vector_db_config=vector_db_config,
                keyword_index_config=keyword_index_config,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                enable_grade_filter=enable_grade_filter,
                enable_mock_data=enable_mock_data,
                content_filter_config=content_filter_config
            )
            
            logger.info("知识检索器实例创建成功")
        except Exception as e:
            logger.error(f"创建知识检索器实例失败: {str(e)}", exc_info=True)
            # 如果创建失败，返回一个基本的检索器实例
            get_retriever.instance = KnowledgeRetriever(
                enable_mock_data=True
            )
            logger.warning("已创建一个基本的知识检索器作为回退方案")
    
    return get_retriever.instance

@app.post("/api/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    retriever: KnowledgeRetriever = Depends(get_retriever)
) -> SearchResponse:
    """
    知识检索API
    
    根据查询文本和过滤条件，检索符合要求的知识项
    """
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 调用检索器执行检索
        results = retriever.retrieve(
            query=request.query,
            grade_level=request.grade_level,
            subject=request.subject,
            top_k=request.top_k,
            min_score=request.min_score,
            apply_curriculum_filter=request.apply_curriculum_filter
        )
        
        # 计算检索耗时
        search_time = time.time() - start_time
        
        # 构造响应
        response = SearchResponse(
            query=request.query,
            grade_level=request.grade_level,
            subject=request.subject,
            search_time=search_time,
            total=len(results),
            items=[
                {
                    "id": result.item.id,
                    "title": result.item.title,
                    "content": result.item.content,
                    "grade_level": result.item.grade_level,
                    "subject": result.item.subject,
                    "keywords": result.item.keywords,
                    "source": result.item.source,
                    "score": result.score,
                    "metadata": getattr(result.item, "metadata", None)
                }
                for result in results
            ]
        )
        
        return response
    except Exception as e:
        logger.error(f"知识检索失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"知识检索失败: {str(e)}"
        ) 