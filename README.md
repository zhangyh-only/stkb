# STKB 方案与能力验证工程

STKB 面向销售训练知识的持续设计与验证。本仓库的主线仍然是方案探索；代码用于证明关键设计能运行、数据能落地、结果能观察，而不是提前建设完整业务系统。

## 技术栈

- Python 3.12 + FastAPI：方案验证接口与处理流程
- Vue 3 + TypeScript + Vite：验证操作、过程和结果展示
- PostgreSQL 16 + pgvector：业务登记、运行记录和向量检索投影
- Neo4j 5：知识图谱投影与关系查询
- 本地文件系统：正式知识 Markdown 形态验证

PostgreSQL 是业务登记和运行记录的事实主存。需要验证的三种知识形态是：正式知识文件、pgvector 检索投影、Neo4j 图投影；三者必须能回到同一知识对象和修订。

## 当前阶段

本轮只建立干净的项目结构、协作规范、基础服务、数据库环境和可视化底座。暂不实现具体资料解析、知识识别、归并或检索算法。

后续每个方案切片都应形成：

```text
方案假设 -> 输入样例 -> Python 实现 -> 三种形态数据 -> Web 展示 -> 对比结论
```

## 目录

```text
services/      Python 验证服务底座
web/           Vue 验证工作台底座
infra/         PostgreSQL/pgvector 与 Neo4j 本地环境
contracts/     经过验证后冻结的跨模块合同
docs/current/  当前领域方案、映射矩阵和决策台账
docs/          技术基线与开发规范
workspace/     本地正式知识文件输出目录，不提交运行数据
samples/       经评审的脱敏输入和预期结果样例
```

## 快速开始

需要 Docker、uv、Node.js 22.13+ 和 pnpm 11。

```bash
make install
make dev
```

- Web：http://localhost:5173
- API：http://localhost:8000/health
- API 文档：http://localhost:8000/docs
- Neo4j Browser：http://localhost:7474

提交前执行：

```bash
make check
```

项目约束见 [AGENTS.md](./AGENTS.md)、[技术基线](./docs/TECHNICAL_BASELINE.md)和[开发规范](./docs/DEVELOPMENT.md)。
