# 开发规范

## 环境

```bash
cp .env.example .env
make install
make dev
```

`.env` 只在本机使用，不提交。

## 新增方案切片

开始编码前写清：

1. 要验证的方案问题；
2. 可重复使用的真实或脱敏样例，评审后放入 `samples/`；
3. 预期输出和判断标准；
4. PostgreSQL、知识文件、pgvector、Neo4j 和页面分别承担什么；
5. 哪些能力不在本次范围。

验证代码放在 `services/app/features/<capability>/`，页面放在 `web/src/features/<capability>/`。验证通过后，再决定是否把 Schema 或 API 纳入 `contracts/`。

## 配置

- Python 配置统一从 `services/app/core/config.py` 读取。
- 前端 API 地址统一使用 `VITE_API_BASE_URL`。
- Docker Compose 变量统一来自根目录 `.env`。
- 共享标识不得在多个模块重复写字面量。

## 检查

```bash
make check
```

涉及存储时还要执行 `make infra`，并用同一输入核对 PostgreSQL、正式知识文件、pgvector 和 Neo4j 的实际数据。
