# 本地运行工作区

本地验证使用以下受控目录：

- `workspace/source-materials/{document-package-id}/`：原始资料；页面按文件名选择，内部 ID 只用于关联。
- `workspace/documents/{document-package-id}/`：已完成的代理解析结果与 `manifest.json`。
- `workspace/knowledge/`：后续能力验证生成的正式知识 Markdown。

原始资料、解析结果和运行数据均不提交。服务首次读取资料时会扫描 `manifest.json`，将原件路径、原件校验值、全文路径和内部 `DocumentPackage` ID 登记到本地 PostgreSQL。需要长期复现的脱敏输入和预期结果经评审后移入 `samples/`。
