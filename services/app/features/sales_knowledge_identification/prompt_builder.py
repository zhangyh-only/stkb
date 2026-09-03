from __future__ import annotations

import json

from .catalog import KNOWLEDGE_MODULES
from .content_contracts import (
    CONTENT_CONTRACT_BY_MODULE,
    render_content_contracts_for_prompt,
)
from .identity_contracts import IDENTITY_CONTRACT_BY_MODULE
from .models import AtomicClaim, CandidateObjectPlan, DocumentPackage, ModelRequest

PROMPT_VERSION = "sales-identification-v0.12"
SCHEMA_VERSION = "candidate-knowledge-object-v0.7"


def render_planning_contracts_for_prompt() -> str:
    sections: list[str] = []
    for module in KNOWLEDGE_MODULES:
        content = CONTENT_CONTRACT_BY_MODULE[module.code]
        identity = IDENTITY_CONTRACT_BY_MODULE[module.code]
        sections.append(
            "\n".join(
                [
                    f"### {module.code} {module.name}",
                    f"- 含义：{module.meaning}",
                    f"- 顶层对象合同：{', '.join(module.canonical_object_types)}",
                    "- 内部条目（只能放在对象 content 内）："
                    + (", ".join(module.item_types) or "无"),
                    f"- 纳入：{content.inclusion}",
                    f"- 排除：{content.exclusion}",
                    f"- 粒度：{content.granularity}",
                    f"- 身份字段：{', '.join(identity.identity_fields)}",
                    f"- 必须拆分：{identity.different_object_when}",
                ]
            )
        )
    return "\n\n".join(sections)


def _compact_claim_for_planning(claim: AtomicClaim) -> dict[str, object]:
    compact = claim.model_dump(by_alias=True)
    compact["moduleHints"] = []
    for evidence in compact["evidence"]:
        excerpt = evidence.pop("exactQuote", "")
        evidence.pop("sourceText", None)
        evidence["sourceExcerpt"] = (
            excerpt if len(excerpt) <= 240 else excerpt[:240] + "…"
        )
    return compact


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
11. moduleHints 只能填写当前12个模块之一；不确定时留空，禁止自造分类名。

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
    "moduleHints": ["D4.2"],
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


def build_document_object_planning_request(
    document_package: DocumentPackage,
    max_candidates: int,
) -> ModelRequest:
    system_prompt = f"""你是 STKB 全文知识发现与对象规划器。直接阅读全文，规划最多
{max_candidates} 个可独立识别、更新和复用的 KnowledgeObject；不要先把每句话展开成独立对象。

规则：
1. 对象边界由共同业务身份、适用范围、更新生命周期和消费职责决定。参数、步骤、清单成员、
   问答条目、话术变体和评分档位在共享身份时放在对象内部。
2. 每个计划选择一个当前模块和 objectType 合同。objectType 只是字段合同，不是知识层级。
3. claims 只保留形成这些对象所需的来源证据，不做全文逐句复述。相同主题、主体和范围的事实可以
   形成一条主张，但必须分别保留每段原文证据。
4. 每条 claim 使用紧凑数组：[claimId,claimKind,subject,attributes,evidence]；evidence 是一个或多个
   [anchorId,exactQuote,selector]。exactQuote 必须是原文中连续、逐字存在的短引句，禁止使用 ...、…
   或把不连续内容拼成一句；需要多处原文时增加 evidence 项。普通段落的 selector 使用 null。
5. 每个对象计划使用紧凑数组：[planId,title,module,objectType,identityHints,sourceClaimIds]。
6. 每条 claim 至少被一个计划引用；无法形成对象的内容进入 unresolvedItems。不要输出 content、
   domain、对象边界说明、实体、关系或分析过程。
7. 下列内容边界必须保持：
   - 同一产品的多个版本在同一来源维护单元中共同呈现、比较和更新时，形成一个版本矩阵计划，
     在 content 内保留各版本差异；只有版本来自不同维护单元、生效周期或独立更新时才拆分。
   - 同一决策上下文共同发布的多个产品组合形成一个 SALES_STRATEGY 策略集，各方案作为内部
     branch；只有独立发布、审批、生效期或下线生命周期时才拆分计划。
   - FAQ 的问答条目共享维护单元时形成一个 QA_PAIR 对象，但每个问题及答案分别形成 qa claim，
     attributes 至少包含 question 和 answer，计划引用全部 qa claim。
   - 完整流程可以形成一个 process claim，attributes 中用 steps 保存来源明确给出的有序步骤。
8. 只输出合法 JSON。

当前规划合同：
{render_planning_contracts_for_prompt()}

claimKind 只能是：fact、list、process、rule、comparison、customer_signal、method、strategy、
script、objection、qa、term、case、asset、value_proposition、evaluation、benchmark。

输出形态：
{{
  "claims": [["CL1","fact","产品能力",{{}},[["真实anchorId","逐字短引句",null]]]],
  "objectPlans": [["P1","业务可读标题","D1.1","PRODUCT_FACT",
    {{"subject":"主体","versionScope":"版本","factTheme":"主题"}},["CL1"]]],
  "weakSignals": [],
  "unresolvedItems": []
}}"""
    return ModelRequest(
        document_package_id=document_package.document_package_id,
        system_prompt=system_prompt,
        user_prompt=(
            "允许使用的来源锚点：\n"
            + "\n".join(anchor.anchor_id for anchor in document_package.anchors)
            + "\n\n全文 Markdown：\n"
            + document_package.full_markdown
        ),
    )


def build_object_planning_request(
    document_package_id: str,
    claims: list[AtomicClaim],
    max_candidates: int,
) -> ModelRequest:
    claims_payload = []
    for claim in claims:
        claims_payload.append(_compact_claim_for_planning(claim))
    system_prompt = f"""你是 STKB 销售知识对象边界规划器。你会一次看到整份资料所有已核验的
原子主张。此阶段只决定应形成哪些可独立识别、归并、更新和消费的对象，以及每个对象使用哪些
主张；不编写 content，不生成正式 KnowledgeObject ID。

规划规则：
1. 最多输出 {max_candidates} 个计划；上限是防失控保护，不是目标数量。粒度由独立消费、业务身份、
   适用范围与更新生命周期决定，绝不按句子数、claimKind 或模块机械拆分。
2. 必须跨 claimKind 观察同一业务对象：同一产品版本的事实、限制和权益可共同形成完整产品版本对象；
   但产品事实、销售策略、客户异议和完整话术职责不同，必须保持独立并通过关系关联。
3. objectType 只选择 KnowledgeObject 的字段、身份和校验合同，不是模块下的新层级；步骤、条目、
   表达变体和评分档位在共享身份与生命周期时留在对象内部。
4. D1.1 保存供给事实、清单、比较和由事实支撑的价值；D1.2 保存获得、使用或交付供给的业务规则
   与完整流程。销售会话的条件动作不属于履约流程。
5. D2.1 保存稳定画像结构和决策角色；D2.2 保存需求、动机和有证据的行为信号。单次表达不能直接
   升格为稳定画像、心理结论或反应规律。
6. D3.1 保存场景与旅程骨架；D3.2 保存通用方法和条件化策略；D3.3 只保存有正式依据的销售行为
   约束。方法、策略、完整回应和事实保持独立，通过关系关联。
7. D4.1 规范客户问题、异议和交互意图；D4.2 保存标准回答、话术和解释；D4.3 保存案例和引用正式
   知识的赋能资产。客户表达不能混入销售回应，回应中的事实必须引用D1。
8. D5.1 保存评价模型和评分规则；D5.2 保存验证知识、识别、检索或评价机制的基准。单次评分和
   未经评审的运行结果不成为规范知识。
9. 每条主张必须至少进入一个 plan，或在 unresolvedItems 中逐条列明 claimId 与原因；不得静默丢失。
   同一主张只有在确实支撑不同下游职责时才可进入多个计划。
10. 一个计划只有一个主模块。identityHints 只写决定“是否同一对象”的稳定业务要素，不写摘要、
   claimId、锚点、模块码或任意技术字段。
11. 只返回输出形态列出的最小字段，不返回 domain、objectBoundary、classificationBasis、content、
   entityMentions、relations 或解释文字；这些由程序根据已发布合同注入，避免重复规则和输出截断。
   只输出合法 JSON，不输出隐藏推理过程。

当前 D1-D5 / 12个候选模块规划合同：
{render_planning_contracts_for_prompt()}

为避免大型资料输出截断，objectPlans 必须使用紧凑数组，每项依次为
[planId,title,module,objectType,identityHints,sourceClaimIds]，禁止改回字段重复的对象形态。
输出形态：
{{
  "objectPlans": [["P1","业务可读标题","D4.2","STANDARD_SCRIPT",
    {{"communicationGoal":"目标","method":"方法","applicability":"范围"}},["CL1"]]],
  "weakSignals": [{{
    "claimId":"CL9",
    "module":"D2.2",
    "reason":"CL9：仅为待验证信号的具体原因",
    "evidence":["真实anchorId"]
  }}],
  "unresolvedItems": [{{
    "claimId":"CL10",
    "description":"CL10：无法确定对象边界",
    "reason":"具体原因",
    "evidence":["真实anchorId"],
    "module":null
  }}]
}}"""
    return ModelRequest(
        document_package_id=document_package_id,
        system_prompt=system_prompt,
        user_prompt=(
            "以下是整份资料的全部已核验主张。sourceExcerpt 来自已逐字校验引句，"
            "仅为规划压缩显示：\n"
            + json.dumps(
                claims_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ),
    )


def build_content_realization_request(
    document_package_id: str,
    plans: list[CandidateObjectPlan],
    claims: list[AtomicClaim],
    batch_label: str,
) -> ModelRequest:
    claim_by_id = {claim.claim_id: claim for claim in claims}
    compact_claims: dict[str, dict[str, object]] = {}
    for claim in claims:
        compact = claim.model_dump(by_alias=True)
        for evidence in compact["evidence"]:
            evidence.pop("sourceText", None)
        compact_claims[claim.claim_id] = compact
    object_tasks = [
        {
            "plan": plan.model_dump(by_alias=True),
            "verifiedClaims": [
                compact_claims[claim_id]
                for claim_id in plan.source_claim_ids
                if claim_id in claim_by_id
            ],
        }
        for plan in plans
    ]
    system_prompt = f"""你是 STKB 销售知识对象内容编制器。对象边界、分类、身份要素和主张归属
已经由全局规划阶段确定；你只能为计划填充可复用的类型化规范内容，不能新增、删除、合并计划，
也不能改变计划的 module、objectType、identityHints 或 sourceClaimIds。

编制规则：
1. 每个计划必须输出一次，以 planId 回链。content 必须满足对应模块合同，不得只有 summary。
2. 完整吸收计划内所有主张：FAQ 的 items 每条 qa claim 至少对应一项；产品版本对象要覆盖事实、
   限制、适用范围和权益；流程保留顺序与条件；话术保留完整原文。
   每个任务只能使用该任务 verifiedClaims 中的证据，禁止借用同批其他任务的主张或法规依据。
3. 需要保留某条主张的完整原文字段时输出 {{"$verbatimFromClaim":"主张ID"}}，系统会用已核验
   sourceText 替换；只需引用具体禁用表达、术语或短句时输出
   {{"$exactQuoteFromClaim":"主张ID"}}。主张 attributes 已提供精确结构字段时输出
   {{"$attributeFromClaim":{{"claimId":"主张ID","attribute":"responseContext"}}}}；例如 D4.1
   expressions 取 expression，resolutionElements 取 responseContext，禁止用同一个完整主张宏
   混入两者。
   不得把长话术压缩成几十字摘要，也不得用完整 sourceText
   代替一个短语列表项。
4. 不用常识补写来源未提供的事实。合同允许为空的字段可显式给空数组；其余缺失必须忠实说明
   unresolved，而不是编造。
   D1.2 流程的 preconditions 只收录资料明确写出的进入条件；从流程目的推导的常识不是来源事实，
   必须删除并输出空数组。exceptions 只能保留资料明确给出的条件和处理，不得补造自然反应。
5. 资料中的培训谚语、讲师观点和经验性判断必须保留来源立场。TERM 的每个 terms 条目中，
   standardExplanation 可准确转述资料主张，sourceStance 必须说明它是何种来源观点，
   usageBoundary 必须写明不能由此推断什么；不得把“往往/可能”改写成客观必然规律。
   D4.1 CUSTOMER_OBJECTION 的 rootConcernHypotheses 也只能来自资料明确表达的原因、顾虑或研究结论；
   仅凭一句客户异议不得推演“担心涨价、害怕麻烦、希望锁定权益”等心理原因。没有依据时必须输出
   空数组；客户原话 expressions 不得混入销售回复。
6. entityMentions 只记录会参与对象身份、过滤或关系查询的稳定业务实体；不得只输出字符串。每项必须
   严格为 {{"mentionId":"P1-M1","text":"原文实体名","proposedType":"PRODUCT",
   "referenceRole":"ABOUT_PRODUCT","sourceRef":"主张中的真实anchorId"}}。不确定类型时不输出。
7. relations 只记录本批明确且有证据的关系，relationType 只能是 ABOUT、APPLIES_TO、
   SUPPORTS、ADDRESSES、GUIDES、CONSTRAINED_BY、EVALUATED_BY、EXEMPLIFIED_BY；每项必须严格为
   {{"relationKind":"entity|object","relationType":"关系类型","sourceRef":"P1或P1-M1",
   "targetRef":"P2或P2-M1","evidence":["真实anchorId"]}}。无法同时满足引用和证据时不输出。
8. sourceClaimIds 只是本任务可用的证据范围，不代表已经写入正文。必须用 claimUsage 逐条声明正文
   实际吸收了哪些主张：claimId 必须来自本任务，role 只能是 primary 或 supporting，contentPaths
   必须指向 content 中真实存在且承载该主张的 JSONPath（如 $.facts[0].description、
   $.items[2].answer、$.script），explanation 用一句话说明该路径如何表达主张。禁止把根级
   factReferences 或来源 ID 列表
   当作正文消费证明；同一主张可以有多个路径，但不能指向空值。
   content 中每个非空业务叶子字段都必须有精确 claimUsage 路径；路径必须落到字符串、数字或布尔值，
   不能用 $.facts[0]、$.items 之类父级对象一次覆盖多项内容。来源没有支持的步骤、适用场景、
   限制、心理原因或行动建议必须删除或按合同留空，不得为了填满合同而推演。
   以下字段属于正式化门禁，任何非空叶子缺少精确 claimUsage 都会保留为调试候选、但禁止形成正式
   KnowledgeObject：D1.1 的 facts、limitations；D1.2 的 preconditions、rulesOrSteps 正文、
   exceptions；D3.2 的 triggerConditions、decisionLogic、actions；D4.2 的 script 和问答；
   D4.1 的 expressions、resolutionElements 及非空 rootConcernHypotheses。
   不要用 summary 或笼统解释代替这些字段的逐项追溯。
9. 计划内没有真实写入正文的主张必须进入 omittedClaims，逐条给出 claimId 与具体业务原因；
   不能既出现在 claimUsage 又出现在 omittedClaims，不能为了覆盖率虚报已消费。
10. 只输出合法 JSON，不输出额外字段。

当前本批对象内容合同：
{render_content_contracts_for_prompt({plan.module for plan in plans})}

输出形态：
{{"realizations":[{{"planId":"P1","content":{{}},"claimUsage":[{{"claimId":"CL1",
"role":"primary","contentPaths":["$.facts[0].description"],"explanation":"该事实已写入事实条目"}}],
"omittedClaims":[{{"claimId":"CL2","reason":"与对象身份不同，应另行规划"}}],
"entityMentions":[],"relations":[]}}]}}"""
    return ModelRequest(
        document_package_id=document_package_id,
        system_prompt=system_prompt,
        user_prompt=(
            f"内容编制批次：{batch_label}\n对象任务（每项计划与证据严格隔离）：\n"
            + json.dumps(object_tasks, ensure_ascii=False, indent=2)
        ),
    )


def build_plan_coverage_repair_request(
    document_package_id: str,
    existing_plans: list[CandidateObjectPlan],
    uncovered_claims: list[AtomicClaim],
) -> ModelRequest:
    compact_plans = [
        {
            "planId": plan.plan_id,
            "title": plan.title,
            "module": plan.module,
            "objectType": plan.object_type,
            "identityHints": plan.identity_hints,
            "sourceClaimIds": plan.source_claim_ids,
        }
        for plan in existing_plans
    ]
    compact_claims = []
    for claim in uncovered_claims:
        compact_claims.append(_compact_claim_for_planning(claim))
    system_prompt = f"""你是 STKB 对象规划覆盖审查器。首轮全局规划后仍有原子主张未被对象计划消费。
你必须逐条审查这些主张，并选择：补入已有对象、新建独立对象、保留为弱信号，或明确列为未决项。

修复规则：
1. 每条未覆盖主张必须且只能由 planAugmentations、objectPlans、weakSignals、unresolvedItems
   至少一种方式明确处理，禁止再次静默遗漏。
2. 主张若只是已有对象的事实、限制、适用范围、问答条目或支撑证据，使用 planAugmentations；
   不能为了提高覆盖率创建重复对象。
3. 主张具有独立消费职责、身份或更新生命周期时才新建 objectPlans。新计划 ID 使用 R1、R2……，
   只输出规划字段，不输出 content、domain、边界、分类依据、实体或关系。
4. 完整话术与方法/策略职责不同；FAQ 的遗漏条目补入已有 QA_PAIR；方法和策略归 D3.2，
   完整回应归 D4.2。
5. moduleHints 已被清空，因为发现阶段提示不是分类结论。必须依据主张语义和当前规则重新裁决。
6. 不得补造来源没有的事实、心理原因、流程步骤或话术。只输出合法 JSON。

当前 D1-D5 / 12个候选模块规划合同：
{render_planning_contracts_for_prompt()}

为避免输出截断，objectPlans 必须使用紧凑数组，每项依次为
[planId,title,module,objectType,identityHints,sourceClaimIds]。
输出形态：
{{
  "planAugmentations": [{{"planId":"P1","sourceClaimIds":["CL9"]}}],
  "objectPlans": [["R1","业务可读标题","D3.2","SALES_TECHNIQUE",
    {{"techniqueName":"名称","purpose":"目的","mechanism":"机制"}},["CL10"]]],
  "weakSignals": [],
  "unresolvedItems": []
}}"""
    return ModelRequest(
        document_package_id=document_package_id,
        system_prompt=system_prompt,
        user_prompt=(
            "首轮已接受对象计划：\n"
            + json.dumps(
                compact_plans,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n\n尚未覆盖的已核验主张：\n"
            + json.dumps(
                compact_claims,
                ensure_ascii=False,
                separators=(",", ":"),
            )
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
