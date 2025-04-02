#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能教育助手系统启动脚本

该脚本是项目的主入口点，用于启动知识服务API、推荐引擎API、用户服务API和课程体系API。
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='智能教育助手系统服务启动脚本')
    
    parser.add_argument(
        '--host',
        type=str,
        default=os.getenv('API_HOST', '0.0.0.0'),
        help='API服务主机地址 (默认: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=int(os.getenv('API_PORT', '8000')),
        help='API服务基础端口 (默认: 8000)'
    )
    
    parser.add_argument(
        '--service',
        type=str,
        choices=['knowledge', 'recommendation', 'user', 'curriculum', 'all'],
        default=os.getenv('SERVICE', 'all'),
        help='要启动的服务 (默认: all)'
    )
    
    return parser.parse_args()


def start_knowledge_service(host, port):
    """启动知识服务API"""
    try:
        from backend.knowledge_service import start_server
        logger.info(f"正在启动知识服务API，地址: {host}:{port}")
        start_server(host=host, port=port)
    except Exception as e:
        logger.error(f"启动知识服务失败: {str(e)}", exc_info=True)
        sys.exit(1)


def start_recommendation_service(host, port):
    """启动推荐引擎API"""
    try:
        from backend.recommendation_api import start_server
        recommendation_port = port + 1  # 推荐服务使用知识服务端口+1
        logger.info(f"正在启动推荐引擎API，地址: {host}:{recommendation_port}")
        start_server(host=host, port=recommendation_port)
    except Exception as e:
        logger.error(f"启动推荐引擎服务失败: {str(e)}", exc_info=True)
        sys.exit(1)


def start_user_service(host, port):
    """启动用户服务API"""
    try:
        from backend.user_service.api import start_server
        user_port = port + 2  # 用户服务使用知识服务端口+2
        logger.info(f"正在启动用户服务API，地址: {host}:{user_port}")
        start_server(host=host, port=user_port)
    except Exception as e:
        logger.error(f"启动用户服务失败: {str(e)}", exc_info=True)
        sys.exit(1)


def start_curriculum_service(host, port):
    """启动课程体系API"""
    try:
        from backend.curriculum_system.api import start_server
        curriculum_port = port + 3  # 课程体系服务使用知识服务端口+3
        logger.info(f"正在启动课程体系API，地址: {host}:{curriculum_port}")
        start_server(host=host, port=curriculum_port)
    except Exception as e:
        logger.error(f"启动课程体系服务失败: {str(e)}", exc_info=True)
        sys.exit(1)


def start_all_services(host, port):
    """启动所有服务"""
    import threading
    
    # 创建服务线程
    knowledge_thread = threading.Thread(
        target=start_knowledge_service,
        args=(host, port),
        daemon=True
    )
    
    recommendation_thread = threading.Thread(
        target=start_recommendation_service,
        args=(host, port),
        daemon=True
    )
    
    user_thread = threading.Thread(
        target=start_user_service,
        args=(host, port),
        daemon=True
    )
    
    curriculum_thread = threading.Thread(
        target=start_curriculum_service,
        args=(host, port),
        daemon=True
    )
    
    # 启动所有服务线程
    knowledge_thread.start()
    recommendation_thread.start()
    user_thread.start()
    curriculum_thread.start()
    
    # 等待所有线程结束(实际不会结束，除非有异常)
    try:
        knowledge_thread.join()
        recommendation_thread.join()
        user_thread.join()
        curriculum_thread.join()
    except KeyboardInterrupt:
        logger.info("接收到终止信号，正在关闭所有服务...")
        sys.exit(0)


def main():
    """主函数"""
    args = parse_arguments()
    
    # 打印启动信息
    logger.info("=" * 50)
    logger.info("智能教育助手系统")
    logger.info("=" * 50)
    logger.info(f"服务: {args.service}")
    logger.info(f"地址: {args.host}:{args.port}")
    
    # 根据选择的服务启动相应的API
    if args.service == 'knowledge':
        start_knowledge_service(args.host, args.port)
    elif args.service == 'recommendation':
        start_recommendation_service(args.host, args.port)
    elif args.service == 'user':
        start_user_service(args.host, args.port)
    elif args.service == 'curriculum':
        start_curriculum_service(args.host, args.port)
    elif args.service == 'all':
        start_all_services(args.host, args.port)
    else:
        logger.error(f"未知服务: {args.service}")
        sys.exit(1)


if __name__ == "__main__":
    main() 