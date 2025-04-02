# 智能教育助手系统 - 集成测试

本目录包含智能教育助手系统的集成测试框架和测试用例，用于确保系统各模块能够协同工作。

## 测试结构

- `integration_tests.py`: 核心集成测试，测试各模块直接交互
- `test_integration_api.py`: API层集成测试，测试各服务API的交互
- `integration_config.py`: 测试配置和数据生成
- `integration_utils.py`: 测试辅助工具和模拟对象
- `run_integration_tests.py`: 测试运行器

## 测试覆盖范围

集成测试涵盖以下核心功能的集成:

1. **知识服务集成**: 知识检索与内容生成
2. **推荐服务集成**: 学生模型与推荐引擎
3. **用户服务集成**: 用户管理与认证
4. **跨服务集成**: 模拟完整用户流程

## 运行测试

### 前提条件

确保已安装所有依赖:

```bash
pip install -r requirements.txt
```

### 运行所有集成测试

```bash
python tests/run_integration_tests.py
```

### 只运行特定类型的测试

```bash
# 只运行核心组件集成测试
python tests/run_integration_tests.py --test-type core

# 只运行API集成测试
python tests/run_integration_tests.py --test-type api
```

### 生成测试报告

```bash
# 生成文本报告
python tests/run_integration_tests.py --output test_report.txt

# 生成XML报告
python tests/run_integration_tests.py --format xml --output test-reports

# 生成HTML报告
python tests/run_integration_tests.py --format html --output test-reports
```

### 更多输出信息

```bash
# 增加详细程度
python tests/run_integration_tests.py -v
python tests/run_integration_tests.py -vv  # 更详细
```

### 调试测试

```bash
# 保留测试数据（不清理）
python tests/run_integration_tests.py --keep-data
```

## 添加新的测试

要添加新的集成测试，请遵循以下步骤:

1. 在合适的测试文件中添加新的测试方法，或创建新的测试类
2. 确保测试方法名以`test_`开头
3. 使用`IntegrationTestSuite`作为基类，以便自动处理测试环境设置和清理
4. 使用`verify_*`辅助方法进行验证
5. 将新的测试添加到相应的测试套件中

## 模拟组件

集成测试使用以下模拟组件，避免依赖外部服务:

- `MockVectorDB`: 模拟向量数据库(FAISS)
- `MockKeywordIndex`: 模拟关键词索引(Elasticsearch)
- 模拟的OpenAI API调用

## 最佳实践

- 尽量使用模拟对象，避免依赖实际的外部服务
- 测试应关注模块之间的交互，而不是单个模块的内部逻辑
- 使用`logger`记录有用的调试信息
- 测试失败时，检查测试目录下的测试数据
- 测试开发完成后，执行带覆盖率报告的测试，确保充分的测试覆盖 