from __future__ import annotations

import json

from .catalog import render_catalog_for_prompt
from .content_contracts import render_content_contracts_for_prompt
from .models import AtomicClaim, DocumentPackage, ModelRequest

PROMPT_VERSION = "sales-identification-v0.5"
SCHEMA_VERSION = "candidate-knowledge-object-v0.4"


def build_claim_discovery_request(
    document_package: DocumentPackage,
    segment_label: str | None = None,
) -> ModelRequest:
    system_prompt = """你是 STKB 销售知识的原子主张发现器。此阶段只负责高召回发现，
不形成 KnowledgeObject、不做跨资料归并，也不把多条独立内容压成摘要。

发现规则：
1. 原子主张是可由明确原文证据支持、后续可以独立判断归类和对象边界的最小业务陈述。
2. 对事实参数、清单、流程步骤、政策规则、方法、策略、完整话术、客户异议、问答、术语、
   案例、物料、价值主张、评估标准分别发现；同一来源可产生多个不同 claimKind 的主张。
3. 完整话术与其体现的策略是两个消费职责：原文同时提供时，分别输出 script 和 strategy。
4. 不同产品版本、客群、产品组合、根本异议、问答条目默认分别发现，不能用“若干”“常见”
   或一个汇总陈述代替。
5. evidence.exactQuote 必须逐字复制自对应来源锚点，不能改写。引句应足以证明该主张，
   但不必复制整段长话术；禁止用省略号替换原文、改换标点或拼接不连续句子。
6. Excel 代理 Markdown 中如证据来自某列，selector 必须填写原列标签（如 B列、C列、D列）；
   普通段落不填 selector。系统会用 selector 回填并保存该字段完整原文。
7. moduleHints 只是候选模块提示，可为多个或为空，不在本阶段强行裁决。
8. attributes 保存后续对象化所需的简短结构信息，不复制长篇原文；不得编造资料未提供的信息。
9. 只输出合法 JSON 对象，不要代码围栏、解释或隐藏推理过程。
10. 逐个来源锚点检查，不得只处理分段后半部分：
    - FAQ 表中同时有问题列和解答列的每一行，至少输出一条 qa；解答含独立业务规则时再输出 rule。
    - 产品话术表中有完整话术列的每一行，至少输出一条 script。
    - 营销行的引入思路/适用客群/产品组合具有独立判断逻辑时，再输出 strategy。
    - 异议处理行还必须输出 objection；异议根因与完整处理话术是两个不同消费职责。
11. moduleHints 只能填写 D1.1 到 D5.4 范围内的模块码；不确定时留空，禁止自造英文分类名。

claimKind 只能是：fact、list、process、rule、comparison、customer_signal、method、strategy、
script、objection、qa、term、case、asset、value_proposition、evaluation、benchmark。

输出形态：
{
  "claims": [{
    "claimId": "CL1",
    "claimKind": "script",
    "statement": "面向忙碌上班族的药享保完整引入话术",
    "subject": "忙碌上班族药享保引入",
    "attributes": {"communicationGoal": "痛点引入", "audience": "忙碌上班族"},
    "moduleHints": ["D4.1"],
    "evidence": [{
      "anchorId": "文档包锚点",
      "exactQuote": "逐字原文短引句",
      "selector": "D列"
    }]
  }]
}"""
    anchors = "\n".join(anchor.anchor_id for anchor in document_package.anchors)
    segment_context = f"技术分段：{segment_label}\n" if segment_label else ""
    return ModelRequest(
        document_package_id=document_package.document_package_id,
        system_prompt=system_prompt,
        user_prompt=f"""DocumentPackage: {document_package.document_package_id}
{segment_context}处理方式: {document_package.processing_method}
已知解析质量问题: {document_package.quality_issues}

允许使用的来源锚点：
{anchors}

待发现原子主张的 Markdown：
{document_package.full_markdown}
""",
    )


def build_object_formation_request(
    document_package_id: str,
    claims: list[AtomicClaim],
    group_label: str,
    max_candidates: int,
) -> ModelRequest:
    claims_payload = [claim.model_dump(by_alias=True) for claim in claims]
    system_prompt = f"""你是 STKB 销售知识对象形成器。输入是已经过程序核验的原子主张，
本阶段依据 D1-D5/22 模块规则、对象粒度和内容合同形成候选 KnowledgeObject。

对象化规则：
1. 最多输出 {max_candidates} 个候选，但不得为了少输出而合并业务身份、独立检索职责或更新
   生命周期不同的对象；超出上限时将未覆盖主张写入 unresolvedItems。
2. 粒度由下游独立消费决定：不同产品版本、客群策略、产品组合、根本异议和完整话术通常
   独立成对象；同一主题下共同更新的 FAQ 条目可形成一个对象，items 必须完整覆盖输入条目。
3. 每个候选必须列出 sourceClaimIds。系统将由这些主张确定正式来源锚点，不接受模型自行
   发明证据。
4. content 必须满足所选模块内容合同，不能只有 summary，也不能用空值、常识或泛化描述补齐。
5. 需要保留某条主张的完整原文字段时，在 content 对应位置输出
   {{"$verbatimFromClaim": "CL1"}}；系统会替换为经过证据校验的 sourceText，避免复制时改写或截断。
6. 一个候选只有一个主模块；跨模块复用通过独立对象和关系表达，不用一个对象兼任多个职责。
7. classificationBasis 只说明命中的规则与边界，objectBoundary 说明共同更新和复用的边界，
   不输出隐藏推理过程。
8. 不生成正式 KnowledgeObject ID，不执行跨资料归并，只输出合法 JSON 对象。

当前 D1-D5 / 22个知识内容模块规则：
{render_catalog_for_prompt()}

当前22个模块对象内容合同：
{render_content_contracts_for_prompt()}

输出形态：
{{
  "candidates": [{{
    "candidateId": "C1",
    "title": "业务可读标题",
    "domain": "D4",
    "module": "D4.1",
    "objectType": "STANDARD_SCRIPT",
    "objectBoundary": "同一沟通目标、客群与更新生命周期下独立复用",
    "classificationBasis": "符合 D4.1 标准表达与完整话术边界",
    "identityHints": {{"subject": "业务主体", "scope": "适用范围"}},
    "sourceClaimIds": ["CL1"],
    "content": {{
      "communicationGoal": "沟通目标",
      "applicability": {{"audience": "适用客群"}},
      "script": {{"$verbatimFromClaim": "CL1"}},
      "factReferences": [],
      "complianceConstraints": []
    }},
    "entityMentions": [],
    "relations": []
  }}],
  "weakSignals": [],
  "unresolvedItems": []
}}"""
    return ModelRequest(
        document_package_id=document_package_id,
        system_prompt=system_prompt,
        user_prompt=(
            f"对象化批次：{group_label}\n"
            "以下 JSON 中 sourceText 是由程序从来源锚点/列选择器回填的完整原文，"
            "exactQuote 已逐字校验：\n"
            + json.dumps(claims_payload, ensure_ascii=False, indent=2)
        ),
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
