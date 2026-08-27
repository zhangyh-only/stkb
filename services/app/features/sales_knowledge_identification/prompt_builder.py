from __future__ import annotations

import json

from .catalog import render_catalog_for_prompt
from .content_contracts import render_content_contracts_for_prompt
from .identity_contracts import render_identity_contracts_for_prompt
from .models import AtomicClaim, CandidateObjectPlan, DocumentPackage, ModelRequest

PROMPT_VERSION = "sales-identification-v0.6"
SCHEMA_VERSION = "candidate-knowledge-object-v0.5"


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
5. evidence.exactQuote 必须逐字复制自对应来源锚点，不能把多处原文拼接或改写。引句应足以证明该主张，
   但不必复制整段长话术；禁止用省略号替换原文、改换标点或拼接不连续句子。
6. Excel 代理 Markdown 中如证据来自某列，selector 必须填写原列标签（如 B列、C列、D列）；
   普通段落不填 selector。系统会用 selector 回填并保存该字段完整原文。
7. moduleHints 只是候选模块提示，可为多个或为空，不在本阶段强行裁决。
8. attributes 保存后续对象化所需的简短结构信息，不复制长篇原文；不得编造资料未提供的信息。
9. 只输出合法 JSON 对象，不要代码围栏、解释或隐藏推理过程。
10. 逐个来源锚点检查，不得只处理分段后半部分；表格中空白的分组/类别单元格继承上方最近的非空值：
    - FAQ 表中同时有问题列和解答列的每一行，至少输出一条 qa；解答含独立业务规则时再输出 rule。
    - 产品话术表中有完整话术列的每一行，至少输出一条 script。
    - 营销行的引入思路/适用客群/产品组合具有独立判断逻辑时，再输出 strategy。
    - 异议处理行还必须输出 objection；异议根因与完整处理话术是两个不同消费职责。
    - 出现“某版本引入/某产品引入”及其保障、价格、权益等说明时，除 script/strategy 外还要输出
      对应 product/version fact，不能遗漏产品版本事实。
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


def build_object_planning_request(
    document_package_id: str,
    claims: list[AtomicClaim],
    max_candidates: int,
) -> ModelRequest:
    claims_payload = []
    for claim in claims:
        compact = claim.model_dump(by_alias=True)
        for evidence in compact["evidence"]:
            evidence.pop("sourceText", None)
        claims_payload.append(compact)
    system_prompt = f"""你是 STKB 销售知识对象边界规划器。你会一次看到整份资料所有已核验的
原子主张。此阶段只决定应形成哪些可独立识别、归并、更新和消费的对象，以及每个对象使用哪些
主张；不编写 content，不生成正式 KnowledgeObject ID。

规划规则：
1. 最多输出 {max_candidates} 个计划；上限是防失控保护，不是目标数量。粒度由独立消费、业务身份、
   适用范围与更新生命周期决定，绝不按句子数、claimKind 或模块机械拆分。
2. 必须跨 claimKind 观察同一业务对象：同一产品版本的事实、限制和权益可共同形成完整产品版本对象；
   但产品事实、销售策略、客户异议和完整话术职责不同，必须保持独立并通过关系关联。
3. FAQ 表按一个共同维护的问答集合规划，所有 qa 主张进入同一计划；不要按每个问题或主题拆对象。
4. 同一沟通目标、客群、产品组合与方法下的完整话术形成一个话术对象；措辞变体进入同一对象。
5. 根本顾虑相同的异议表达归并；根本顾虑不同的异议保持独立。异议对象不吞并应对话术。
6. 下列边界是硬约束：
   - 每个不同 productCombination/产品组合形成独立 D3.3 策略，不得因同属组合营销而合并；
   - D1.3 的用户操作顺序形成 BUSINESS_PROCESS；处方限量、同功效药限制、目录限制、发票条件等
     可独立查询的约束形成 POLICY_RULE_SET，不得塞进一个“通用流程与规则”对象；
   - 同一规则主题可聚合多条约束，但处方规则与发票规则因消费问题和更新依据不同必须拆分。
7. 每条主张必须至少进入一个 plan，或在 unresolvedItems 中逐条列明 claimId 与原因；不得静默丢失。
   同一主张只有在确实支撑不同下游职责时才可进入多个计划。
8. 一个计划只有一个主模块。identityHints 只写决定“是否同一对象”的稳定业务要素，不写摘要、
   claimId、锚点、模块码或任意技术字段。
9. 只返回输出形态列出的最小字段，不返回 domain、objectBoundary、classificationBasis、content、
   entityMentions、relations 或解释文字；这些由程序根据已发布合同注入，避免重复规则和输出截断。
   只输出合法 JSON，不输出隐藏推理过程。

当前 D1-D5 / 22个知识内容模块规则：
{render_catalog_for_prompt()}

当前22个模块对象粒度、纳入/排除与正反例：
{render_content_contracts_for_prompt()}

当前22个模块对象身份与归并合同：
{render_identity_contracts_for_prompt()}

输出形态：
{{
  "objectPlans": [{{
    "planId": "P1",
    "title": "业务可读标题",
    "module": "D4.1",
    "objectType": "STANDARD_SCRIPT",
    "identityHints": {{"subject": "业务主体", "scope": "适用范围"}},
    "sourceClaimIds": ["CL1"]
  }}],
  "weakSignals": [],
  "unresolvedItems": [{{"description":"CL9：无法确定对象边界", "reason":"具体原因", "evidence":[]}}]
}}"""
    return ModelRequest(
        document_package_id=document_package_id,
        system_prompt=system_prompt,
        user_prompt=(
            "以下是整份资料的全部已核验主张。exactQuote 已逐字校验：\n"
            + json.dumps(claims_payload, ensure_ascii=False, indent=2)
        ),
    )


def build_content_realization_request(
    document_package_id: str,
    plans: list[CandidateObjectPlan],
    claims: list[AtomicClaim],
    batch_label: str,
) -> ModelRequest:
    plan_payload = [plan.model_dump(by_alias=True) for plan in plans]
    claim_payload = [claim.model_dump(by_alias=True) for claim in claims]
    system_prompt = f"""你是 STKB 销售知识对象内容编制器。对象边界、分类、身份要素和主张归属
已经由全局规划阶段确定；你只能为计划填充可复用的类型化规范内容，不能新增、删除、合并计划，
也不能改变计划的 module、objectType、identityHints 或 sourceClaimIds。

编制规则：
1. 每个计划必须输出一次，以 planId 回链。content 必须满足对应模块合同，不得只有 summary。
2. 完整吸收计划内所有主张：FAQ 的 items 每条 qa claim 至少对应一项；产品版本对象要覆盖事实、
   限制、适用范围和权益；流程保留顺序与条件；话术保留完整原文。
3. 需要保留某条主张的完整原文字段时输出 {{"$verbatimFromClaim":"主张ID"}}，系统会用已核验
   sourceText 替换。不得把长话术压缩成几十字摘要。
4. 不用常识补写来源未提供的事实。合同允许为空的字段可显式给空数组；其余缺失必须忠实说明
   unresolved，而不是编造。
5. entityMentions 只记录会参与对象身份、过滤或关系查询的稳定业务实体；不得只输出字符串。每项必须
   严格为 {{"mentionId":"P1-M1","text":"原文实体名","proposedType":"PRODUCT",
   "referenceRole":"ABOUT_PRODUCT","sourceRef":"主张中的真实anchorId"}}。不确定类型时不输出。
6. relations 只记录本批明确且有证据的关系，每项必须严格为
   {{"relationKind":"entity|object","relationType":"关系类型","sourceRef":"P1或P1-M1",
   "targetRef":"P2或P2-M1","evidence":["真实anchorId"]}}。无法同时满足引用和证据时不输出。
7. 只输出合法 JSON，不输出额外字段。

当前22个模块对象内容合同：
{render_content_contracts_for_prompt()}

输出形态：
{{"realizations":[{{"planId":"P1","content":{{}},"entityMentions":[],"relations":[]}}]}}"""
    return ModelRequest(
        document_package_id=document_package_id,
        system_prompt=system_prompt,
        user_prompt=(
            f"内容编制批次：{batch_label}\n对象计划：\n"
            + json.dumps(plan_payload, ensure_ascii=False, indent=2)
            + "\n计划引用的已核验主张（sourceText 为程序回填的完整来源字段）：\n"
            + json.dumps(claim_payload, ensure_ascii=False, indent=2)
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
