# 智能教育助手系统

## 项目简介

智能教育助手系统是一个基于大模型和多智能体技术的K12教育辅导平台，旨在为学生提供个性化、符合课标的学习体验。系统结合了知识检索、个性化推荐、作业批改和多角色协作教学等功能，为学生打造全方位的学习支持环境。

## 项目当前状态

✅ 已完成知识服务模块核心实现
✅ 已完成向量数据库(FAISS)和关键词索引(Elasticsearch)集成
✅ 已完成个性化练习推荐引擎基础框架实现
✅ 已完成课标知识体系构建和基于课标的内容过滤机制
✅ 已完成内容质量评估机制优化
✅ 已完成用户画像与学习记录整合
🔄 正在进行课标知识点数据收集与导入
📅 计划下一步实现多智能体协作框架和作业批改智能体

## 核心特色

- **严格遵循课标**：所有内容生成和知识检索严格符合各年级课标要求
- **个性化学习路径**：根据学生学习历史和掌握程度动态调整内容难度
- **记忆科学支持**：基于艾宾浩斯遗忘曲线模型，智能安排复习内容
- **多智能体协作**：模拟教师、助教等角色，提供多视角指导
- **高效资源利用**：采用轻量化技术和动态缓存机制，实现高性能低成本运行
- **智能内容过滤**：确保所有内容严格符合课标要求，自动识别并过滤超纲内容
- **全面质量评估**：多维度评估内容质量，自动生成改进建议
- **学习数据分析**：深入分析学习历史，提供个性化学习建议

## 项目文档

- [项目策划讨论](./智能教育助手系统-项目策划.md)：详细的项目设计和规划
- [项目开发进度](./项目开发进度.md)：当前开发状态和待办事项列表
- [项目进度备忘录](./项目进度备忘录.md)：详细的进度跟踪和技术问题记录

## 技术架构

本项目采用前后端分离架构，结合大模型API和多智能体框架，主要技术栈包括：

- **前端**：响应式Web应用，支持多端访问
- **后端**：微服务架构，支持弹性扩展
  - 知识服务模块：基于FastAPI的知识检索和内容生成服务
  - 推荐服务模块：基于学生模型的个性化练习推荐服务
  - 用户服务模块：用户管理、认证和学习记录整合
  - 课标体系模块：管理课标知识点和知识点间关系
  - 内容过滤模块：确保内容符合课标要求
  - 内容质量评估模块：多维度评估内容质量
- **AI模型**：大语言模型、推荐系统、评估模型等
- **数据存储**：
  - 向量数据库(FAISS)：高效存储和检索语义向量
  - 全文索引(Elasticsearch)：实现关键词搜索
  - 知识图谱(NetworkX)：构建和管理课标知识体系
  - 关系型+非关系型混合存储架构

## 使用场景

1. **个性化学习辅导**：根据学生特点提供针对性知识讲解
2. **智能练习推荐**：基于学生掌握程度和遗忘曲线推荐练习题
3. **智能作业批改**：快速准确评估学生作业，提供改进建议
4. **学习规划指导**：根据学习目标制定合理的学习计划
5. **多维度能力评估**：全面评估学生各方面学习能力
6. **内容适配度评估**：自动评估内容是否符合特定年级和学科的课标要求
7. **学习数据分析**：分析学习历史，识别优势和薄弱知识点

## 开发路线图

1. **第一阶段（已完成）**：基础框架搭建，实现核心知识检索和内容生成
   - ✅ 完成KnowledgeRetriever和ContentGenerator模块基础实现
   - ✅ 设计并实现核心API接口
   - ✅ 建立单元测试框架

2. **第二阶段（已完成）**：数据源集成和推荐引擎开发
   - ✅ 集成FAISS向量数据库
   - ✅ 实现Elasticsearch关键词索引
   - ✅ 设计学生模型和遗忘曲线
   - ✅ 实现个性化练习推荐引擎

3. **第三阶段（已完成）**：用户服务开发和课标知识体系构建
   - ✅ 设计用户数据模型和认证系统
   - ✅ 收集各年级、各学科课标要求
   - ✅ 设计知识点标注和关联机制
   - ✅ 实现课程体系服务层和API

4. **第四阶段（已完成）**：基于课标的内容过滤
   - ✅ 设计内容过滤器组件
   - ✅ 实现课标适配度评估逻辑
   - ✅ 开发内容修改建议生成功能
   - ✅ 与知识服务模块集成

5. **第五阶段（已完成）**：系统质量优化与用户体验提升
   - ✅ 完善内容质量评估机制
     - ✅ 设计多维度质量评估模型
     - ✅ 实现8种核心质量维度评估
     - ✅ 与内容生成器集成
     - ✅ 实现改进建议自动生成功能
   - ✅ 实现用户画像与学习记录整合
     - ✅ 设计用户学习历史数据模型
     - ✅ 实现用户画像与学生模型同步机制
     - ✅ 开发学习记录分析与统计功能
     - ✅ 实现知识点掌握度详情查询
     - ✅ 添加个性化学习路径推荐API

6. **第六阶段（进行中）**：课标知识点数据库构建与系统优化
   - 🔄 收集实际课标知识点数据
   - 🔄 构建知识点导入和验证流程
   - 🔄 优化系统性能和资源使用
   - 📅 实现动态缓存机制
   - 📅 优化数据库访问性能

7. **第七阶段（计划中）**：多智能体协作和作业批改
   - 📅 设计并实现多智能体协作框架
   - 📅 实现作业批改智能体
   - 📅 实现动态提示词缓存
   - 📅 研究模型轻量化方案
   - 📅 系统优化与部署

## 快速开始

### 环境要求

- Python 3.8+
- FAISS向量库
- Elasticsearch 8.x
- NetworkX (用于知识图谱)
- PyGraphviz (用于知识图谱可视化，可选)
- OpenAI API密钥（或兼容的API服务）

### 安装

1. 克隆代码库
   ```bash
   git clone https://github.com/yourusername/intelligent-education-assistant.git
   cd intelligent-education-assistant
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 环境配置
   创建`.env`文件并配置以下参数：
   ```
   # OpenAI API配置
   OPENAI_API_KEY=your_api_key
   OPENAI_API_BASE=https://api.openai.com/v1
   
   # 向量数据库配置
   VECTOR_DB_DIMENSION=768
   VECTOR_DB_INDEX_TYPE=Flat
   VECTOR_DB_PATH=./data/vector_db
   
   # Elasticsearch配置
   ES_HOSTS=http://localhost:9200
   ES_INDEX_NAME=knowledge_items
   
   # 推荐引擎配置
   FORGETTING_BASE_RETENTION=0.9
   FORGETTING_RATE=0.1
   PRACTICE_BOOST=0.1
   
   # 课标体系配置
   CURRICULUM_STORAGE_PATH=./data/curriculum
   
   # 内容过滤器配置
   ENABLE_CURRICULUM_FILTER=true
   CURRICULUM_FILTER_STRICTNESS=0.7
   
   # 用户服务配置
   USER_STORAGE_PATH=./data/users
   AUTH_SECRET_KEY=your-secret-key
   TOKEN_EXPIRE_MINUTES=60
   
   # 学习记录整合配置
   LEARNING_SYNC_INTERVAL=3600
   LEARNING_TREND_DAYS=30
   ENABLE_AUTO_SYNC=true
   ```

### 运行

启动所有服务：
```bash
python run.py --service all
```

仅启动知识服务：
```bash
python run.py --service knowledge
```

仅启动推荐服务：
```bash
python run.py --service recommendation
```

仅启动用户服务：
```bash
python run.py --service user
```

仅启动课程体系服务：
```bash
python run.py --service curriculum
```

### API接口

#### 知识服务API (默认端口: 8000)

- `GET /` - 服务信息
- `POST /api/search` - 知识检索
- `POST /api/generate` - 内容生成
- `POST /api/knowledge` - 添加知识项
- `GET /health` - 健康检查

#### 推荐服务API (默认端口: 8001)

- `GET /` - 服务信息
- `POST /api/recommend` - 获取练习推荐
- `POST /api/practice` - 提交练习记录
- `GET /health` - 健康检查

#### 用户服务API (默认端口: 8002)

- `GET /` - 服务信息
- `POST /api/auth/register` - 用户注册
- `POST /api/auth/login` - 用户登录
- `GET /api/users/me` - 获取当前用户信息
- `POST /api/users/{user_id}/practice` - 提交练习记录
- `GET /api/users/{user_id}/learning/stats` - 获取学习统计
- `GET /api/users/{user_id}/learning/knowledge_mastery` - 获取知识点掌握详情
- `GET /api/users/{user_id}/learning/path` - 获取推荐学习路径
- `POST /api/users/{user_id}/learning/sync` - 同步学习记录
- `GET /health` - 健康检查

#### 课程体系API (默认端口: 8003)

- `GET /` - 服务信息
- `POST /api/knowledge_points` - 创建知识点
- `GET /api/knowledge_points/{kp_id}` - 获取知识点
- `POST /api/knowledge_relations` - 创建知识点关系
- `POST /api/filter` - 过滤知识点
- `POST /api/learning_path` - 规划学习路径
- `GET /health` - 健康检查

## 贡献指南

我们欢迎各种形式的贡献，包括但不限于：

- 提交Bug和功能建议
- 改进文档和代码注释
- 增加新功能和修复问题
- 分享使用经验和教学案例

## 许可证

本项目采用 [MIT 许可证](LICENSE)

## 联系方式

- 项目负责人：[待定]
- 邮箱：[待定]
- 官方网站：[待定] 