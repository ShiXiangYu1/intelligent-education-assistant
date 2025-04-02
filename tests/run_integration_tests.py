#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
集成测试运行器 - 智能教育助手系统

此脚本用于运行所有集成测试，收集测试结果并生成报告。
提供命令行参数来控制测试范围、详细程度和报告格式。

作者: AI助手
创建日期: 2023-04-02
"""

import os
import sys
import argparse
import unittest
import logging
import time
from datetime import datetime

# 将项目根目录添加到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入测试模块
from tests.integration_tests import suite as integration_suite
from tests.test_integration_api import suite as api_integration_suite
from tests.integration_utils import setup_test_environment, cleanup_test_environment

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """
    解析命令行参数
    
    返回:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='智能教育助手系统集成测试运行器')
    
    parser.add_argument('--test-type', type=str, default='all',
                       choices=['all', 'core', 'api'],
                       help='要运行的测试类型: all(所有), core(核心组件), api(API集成)')
    
    parser.add_argument('--verbose', '-v', action='count', default=1,
                       help='输出详细程度，可重复使用增加详细程度，如 -v, -vv, -vvv')
    
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='测试报告输出文件路径')
    
    parser.add_argument('--format', '-f', type=str, default='text',
                       choices=['text', 'xml', 'html'],
                       help='测试报告格式: text(文本), xml(XML), html(HTML)')
    
    parser.add_argument('--keep-data', action='store_true',
                       help='保留测试数据（用于调试）')
    
    return parser.parse_args()


def create_test_suite(args):
    """
    根据参数创建测试套件
    
    参数:
        args: 命令行参数
        
    返回:
        测试套件
    """
    suite = unittest.TestSuite()
    
    if args.test_type in ['all', 'core']:
        logger.info("添加核心组件集成测试...")
        suite.addTest(integration_suite())
    
    if args.test_type in ['all', 'api']:
        logger.info("添加API集成测试...")
        suite.addTest(api_integration_suite())
    
    return suite


def run_tests(args, test_suite):
    """
    运行测试套件并处理结果
    
    参数:
        args: 命令行参数
        test_suite: 测试套件
        
    返回:
        测试结果
    """
    if args.format == 'text' or args.output is None:
        # 使用文本测试运行器
        runner = unittest.TextTestRunner(verbosity=args.verbose)
        result = runner.run(test_suite)
    elif args.format == 'xml':
        # 使用XML测试运行器
        import xmlrunner
        if args.output:
            runner = xmlrunner.XMLTestRunner(output=args.output)
        else:
            runner = xmlrunner.XMLTestRunner(output='test-reports')
        result = runner.run(test_suite)
    elif args.format == 'html':
        # 使用HTML测试运行器
        import HtmlTestRunner
        if args.output:
            runner = HtmlTestRunner.HTMLTestRunner(output=args.output, verbosity=args.verbose)
        else:
            runner = HtmlTestRunner.HTMLTestRunner(output='test-reports', verbosity=args.verbose)
        result = runner.run(test_suite)
    
    return result


def generate_report(args, result, start_time, end_time):
    """
    生成测试报告
    
    参数:
        args: 命令行参数
        result: 测试结果
        start_time: 开始时间
        end_time: 结束时间
    """
    if args.format != 'text' or not args.output:
        return
    
    # 计算测试时间
    duration = end_time - start_time
    
    # 生成简单的文本报告
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("=================================\n")
        f.write("智能教育助手系统集成测试报告\n")
        f.write("=================================\n\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试类型: {args.test_type}\n")
        f.write(f"测试耗时: {duration:.2f}秒\n\n")
        
        f.write("测试结果汇总:\n")
        f.write(f"  运行测试数: {result.testsRun}\n")
        f.write(f"  成功数: {result.testsRun - len(result.failures) - len(result.errors)}\n")
        f.write(f"  失败数: {len(result.failures)}\n")
        f.write(f"  错误数: {len(result.errors)}\n\n")
        
        if result.failures:
            f.write("失败的测试:\n")
            for test, err in result.failures:
                f.write(f"  {test}\n")
                f.write(f"  错误信息: {err}\n\n")
        
        if result.errors:
            f.write("发生错误的测试:\n")
            for test, err in result.errors:
                f.write(f"  {test}\n")
                f.write(f"  错误信息: {err}\n\n")
        
        f.write("\n=================================\n")
        f.write("测试报告生成完成\n")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    logger.info(f"开始运行智能教育助手系统集成测试，测试类型: {args.test_type}")
    
    # 设置测试环境
    setup_test_environment()
    
    # 创建测试套件
    test_suite = create_test_suite(args)
    
    # 运行测试
    start_time = time.time()
    result = run_tests(args, test_suite)
    end_time = time.time()
    
    # 生成报告
    if args.output:
        generate_report(args, result, start_time, end_time)
        logger.info(f"测试报告已生成: {args.output}")
    
    # 清理测试环境
    cleanup_test_environment(keep_data=args.keep_data)
    
    # 返回测试结果状态码
    return 0 if (len(result.failures) + len(result.errors)) == 0 else 1


if __name__ == '__main__':
    """
    脚本入口点
    
    使用示例:
    1. 运行所有测试:
       python tests/run_integration_tests.py
    
    2. 只运行API集成测试:
       python tests/run_integration_tests.py --test-type api
    
    3. 生成HTML报告:
       python tests/run_integration_tests.py --format html --output test-reports
    
    4. 详细输出:
       python tests/run_integration_tests.py -vv
    
    5. 保留测试数据:
       python tests/run_integration_tests.py --keep-data
    """
    sys.exit(main()) 