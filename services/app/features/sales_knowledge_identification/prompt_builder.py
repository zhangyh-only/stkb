from .catalog import render_catalog_for_prompt
from .content_contracts import render_content_contracts_for_prompt
from .models import DocumentPackage, ModelRequest

PROMPT_VERSION = "sales-identification-v0.4"
SCHEMA_VERSION = "candidate-knowledge-object-v0.3"


def build_model_request(
    document_package: DocumentPackage,
    max_candidates: int = 10,
    segment_label: str | None = None,
) -> ModelRequest:
    system_prompt = f"""你是 STKB 销售知识识别器。
你需要阅读本次提供的 Markdown 内容并综合识别销售知识，不得按五域或22个知识内容模块循环调用。

识别规则：
1. 输出0到多个候选知识；按可独立归并和更新的业务对象聚合，不按句子、字段或表格单元格机械拆分。
2. 每个候选只有一个主域、主模块和对象类型，但一份文档可以产生跨域候选。
2.1 最多输出 {max_candidates} 个候选；优先合并共享业务身份和更新边界的字段、步骤、
    问答和表达变体，避免为了覆盖模块而拆散对象。
2.2 content 只保留支持对象识别、后续归并和消费所需的必要字段，
    避免重复原文和长篇摘要。
3. 只使用输入中给出的来源锚点；没有明确证据时不要生成候选。
4. 实体提及仍是原文文本，不得创建正式实体 ID。
5. 不生成正式 KnowledgeObject ID，不执行跨资料归并，不输出通用置信度或推理过程。
6. 明确不属于 STKB 的内容不形成候选；无法判断的内容进入 unresolvedItems。
7. 请严格输出 JSON 对象，不要输出 Markdown 代码围栏或解释文字。
8. 分类必须同时依据模块业务含义、允许对象类型、对象边界和消费职责，不能只按关键词、
   资料名称或典型来源判断。
9. 同一证据可以支持多个候选，但只有业务身份、更新边界或消费职责确实独立时才拆分。
10. 多个模块都可能成立且规则不足以裁决时，进入 unresolvedItems，不为填满覆盖矩阵强制分类。
11. 每个候选必须给出可读标题、对象边界、分类依据和非空身份线索。
    身份线索只描述后续归并需要比较的业务维度，不能创建正式 ID。
12. classificationBasis 只说明引用了哪项模块职责和边界，不输出隐藏推理过程；
    objectBoundary 说明为什么这些内容应作为一项对象共同更新和复用。
13. content 必须严格满足所选模块的内容合同，不能只输出 summary；资料未提供必填信息时，
    应缩小对象范围或放入 unresolvedItems，不能用空值、常识或泛化描述补齐。

当前 D1-D5 / 22个知识内容模块识别规则包：
{render_catalog_for_prompt()}

当前22个模块的对象内容合同：
{render_content_contracts_for_prompt()}

输出对象形态：
{{
  "candidates": [{{
    "candidateId": "C1",
    "title": "便于业务人员识别的候选对象标题",
    "domain": "D1",
    "module": "D1.1",
    "objectType": "PRODUCT_FACT",
    "objectBoundary": "共享同一业务身份、适用范围和更新生命周期，因此作为一项对象提议",
    "classificationBasis": "内容属于产品可核验事实，按 D1.1 业务含义和边界归类",
    "identityHints": {{"subject": "原文中的业务主体", "scope": "适用范围或上下文"}},
    "content": {{
      "subject": "原文中的产品或版本主体",
      "facts": [{{"name": "事实名称", "value": "事实值", "evidence": ["文档包锚点"]}}],
      "applicability": {{"scope": "适用范围", "effectiveTime": "原文明确时填写"}},
      "limitations": ["限制与例外"]
    }},
    "entityMentions": [{{
      "mentionId": "M1",
      "text": "原文称呼",
      "proposedType": "PRODUCT",
      "referenceRole": "ABOUT_PRODUCT",
      "sourceRef": "文档包锚点"
    }}],
    "evidence": ["文档包锚点"],
    "relations": []
  }}],
  "weakSignals": [{{"module": "D2.3", "reason": "弱线索原因", "evidence": ["文档包锚点"]}}],
  "unresolvedItems": [{{
    "description": "无法判断内容",
    "reason": "原因",
    "evidence": ["文档包锚点"]
  }}]
}}"""
    anchors = "\n".join(anchor.anchor_id for anchor in document_package.anchors)
    segment_context = (
        f"当前处理文档技术分段: {segment_label}。该分段只为控制上下文，"
        "仍须在本段内对22个知识内容模块综合识别。\n"
        if segment_label
        else ""
    )
    content_label = "Markdown 结构分段" if segment_label else "全文 Markdown"
    user_prompt = f"""DocumentPackage: {document_package.document_package_id}
{segment_context}处理方式: {document_package.processing_method}
已知解析质量问题: {document_package.quality_issues}

允许使用的来源锚点：
{anchors}

{content_label}：
{document_package.full_markdown}
"""
    return ModelRequest(
        document_package_id=document_package.document_package_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def build_repair_request(
    document_package_id: str, raw_output: str, parse_error: str
) -> ModelRequest:
    return ModelRequest(
        document_package_id=document_package_id,
        system_prompt=(
            "你是 JSON 格式修复器。只修复输入的 JSON 语法和顶层结构，不新增、删除或"
            "改写任何销售知识语义。只输出一个合法 JSON 对象，不要输出代码围栏或解释。"
        ),
        user_prompt=(
            f"解析错误：{parse_error}\n\n请将以下模型原始输出修复为合法 JSON：\n{raw_output}"
        ),
    )
