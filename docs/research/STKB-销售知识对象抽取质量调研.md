# STKB 销售知识对象抽取质量调研

> 研究日期：2026-08-27
> 研究目的：为下一轮 STKB 销售域规则、知识对象合同、测试集和调优实验提供一手资料依据。
> 证据口径：外部材料只采用项目官方文档、官方源码仓库或论文原文；对 STKB 的判断标注为“对 STKB 的推断”，不把外部系统的能力直接写成 STKB 已具备的能力。

## 结论先行

当前 STKB 的问题不是“再补几个字段”就能解决，而是把以下四个任务压在了一次模型输出里：

```text
发现原文证据 → 判断是否是销售知识 → 归属销售域/模块 → 聚合为业务对象
```

现有代码的事实是：

- `CandidateKnowledgeObject.content` 是任意非空 `dict`，程序只验证“不为空”，没有按 `object_type` 选择不同的内容合同。见 [models.py](../../services/app/features/sales_knowledge_identification/models.py)。
- 当前 Prompt 的示例核心仍是 `content.summary`，同时要求“避免长篇摘要”。因此模型给出几十字摘要在合同上是合法的，但这不能证明对象内容完整。见 [prompt_builder.py](../../services/app/features/sales_knowledge_identification/prompt_builder.py)。
- `evidence` 目前是锚点 ID 列表，不是“原文摘录 + 在全文中的字符区间 + 该证据支持的字段”。这不足以验证对象字段是否真的由原文支持。
- 长文档按分段并发调用后，结果主要是列表汇总；同一对象跨分段的语义聚合、拆分和冲突判定不应由简单的字符串去重替代。见 [service.py](../../services/app/features/sales_knowledge_identification/service.py)。
- 归并阶段当前按 `object_type + identity_hints` 生成哈希身份；多个候选的内容会被包进 `mergedItems`，这不是面向业务对象的内容整合。见 [formalizer.py](../../services/app/features/sales_knowledge_identification/formalizer.py)。

这些事实带来的推断是：当前“抽出了 10 个对象”不能回答“10 个是否合理”，“每个几十字”也不能回答“是否可用”。对象数量、摘要长度和模块覆盖数只能做诊断信号，不能作为质量结论。

下一轮应把识别改成可验收的纵向质量链：

```text
带精确定位的原文证据
  → 类型化销售知识事实/主张
  → 规则约束下的模块归属与对象边界
  → 对象内容完整性检查
  → 有界的实体归一与跨资料归并
  → 通过硬门槛后才形成正式 KnowledgeObject
```

外部资料最值得借鉴的共同点有三项：

1. 结构化输出只能解决形状，不能解决业务真值；必须同时做语义、证据和业务边界验收。
2. 抽取结果必须能回到精确的来源片段，证据不是装饰性的来源 ID。
3. 生成式抽取不能只用精确匹配的 precision/recall；对象粒度、冗余、完整性和稳定性需要单独测量。

## 一、外部资料对比

| 来源 | 类型 | 一手事实 | 对 STKB 可借鉴的机制 | 不应直接照搬 |
| --- | --- | --- | --- | --- |
| [Google LangExtract](https://github.com/google/langextract) | 官方开源库 | 通过 Prompt/示例进行结构化抽取，把每个抽取值对齐到原文字符区间；支持长文档分段、并行、多轮抽取和交互式原文高亮。[官方 README](https://github.com/google/langextract/blob/main/README.md) | 每个对象字段保存 `source_quote`、`start/end`、锚点和支持字段；把“证据检查”做成可视化审阅 | 不直接引入 Gemini、LangExtract 的抽取类或示例格式；STKB 仍使用现有 OpenAI Compatible Gateway 和自身 D1-D5 合同 |
| [OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) + [JSON Schema](https://json-schema.org/specification) | 官方 API 方案与规范 | `strict: true` 可约束输出匹配开发者提供的 JSON Schema；但官方也明确指出，Schema 合法不代表字段值正确，模型仍可能在值上犯错 | 用 JSON Schema/Pydantic 生成候选合同，结构失败直接阻断；另设语义和证据质量门槛 | 不把“模型返回了合法 JSON”当成识别完成，不依赖并非所有兼容网关都支持的原生 Structured Outputs |
| [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | 官方开源仓库与文档 | 标准索引流按 text unit 抽取实体、关系和 claims，再做实体/关系跨片段汇总、社区发现、分层报告和向量化。[索引概览](https://microsoft.github.io/graphrag/index/overview/) [输出表结构](https://microsoft.github.io/graphrag/index/outputs/) | 保留“文档—文本单元—实体—关系—主张—证据”的中间账本；将跨文档关系构建放在正式对象之后 | 不把 community report、entity summary 或 Parquet 表当作 STKB `KnowledgeObject`；不让图聚类反向决定知识身份 |
| [LlamaIndex PropertyGraphIndex](https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/) | 官方开发文档与源码 | `PropertyGraphIndex` 以 extractor 对每个 chunk 生成节点和关系；`SchemaLLMPathExtractor` 可用 Pydantic、允许的实体/关系类型和严格校验验证路径。[文档源码](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/property_graph/transformations/schema_llm.py) | 借鉴“类型闭集 + 关系端点约束 + 每条路径单独校验”；图查询结果保留来源文本 | 不为当前识别页引入完整框架；不让 Neo4j/LlamaIndex 的节点 ID 取代 STKB 业务对象 ID |
| [GenRES](https://arxiv.org/abs/2402.10744) | 论文原文 | 对生成式关系抽取，单纯硬匹配 precision/recall 不能覆盖输出多样性；提出主题相关性、唯一性、粒度、事实性、完整性等多维评估 | 将对象粒度、冗余、事实支持、覆盖完整性单列为指标；对候选对象做软匹配和人工复核 | 不直接使用其英文关系数据集和分数；STKB 要按中文销售材料定义对象 gold |
| [DocRED](https://aclanthology.org/P19-1074/) / [Revisiting DocRED](https://aclanthology.org/2022.emnlp-main.580/) | 论文与公开数据集 | 文档级关系需要跨句整合，并为关系提供支持证据；后续研究发现推荐—修订式标注会产生大量 false negative | 测试集要覆盖跨页/跨段关系并允许修订 gold；证据标注要独立于对象标签 | 不把 Wikipedia/Wikidata 的实体类型、关系集或分数当成保险销售知识标准 |
| [DREEAM](https://aclanthology.org/2023.eacl-main.145/) | 论文原文 | 证据句可作为文档级关系抽取的监督信号；证据检索本身需要评估 | 先找证据再判断对象，分别测证据召回和关系/对象正确性 | 不直接采用其模型结构；当前阶段先做可解释的证据账本和评估，不做模型训练 |
| [Evidence Attribution 评估](https://aclanthology.org/2025.naacl-long.282/) | 论文原文 | 通过“遮蔽引用—恢复证据”测试归因；研究发现机器生成解释仍会错误归因，人工精选证据更可靠 | 为每项对象主张做证据充分性和可恢复性测试；LLM judge 只能作为辅助 | 不把 LLM judge 的高相关性当成事实验证，不跳过业务复核 |
| [OpenAI Evals](https://github.com/openai/evals) + [Evaluation Flywheel](https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/Building_resilient_prompts_using_an_evaluation_flywheel.md) | 官方开源框架与官方 Cookbook | 评估循环包含失败样例分析、结构化标注、基线测量、自动 grader 和定向改进；同一评估版本应可复现 | 把每轮 Prompt/规则/模型变更与固定测试集、错误分类和回归结果绑定 | 不直接套用通用问答模板；STKB 需要自定义对象、证据、模块和粒度 grader |

## 二、结构化抽取和来源证据

### 2.1 结构化输出的边界

**外部事实。** OpenAI 的 Structured Outputs 文档说明，JSON mode 只提高合法 JSON 的概率，并不保证符合指定 Schema；`strict: true` 的 Structured Outputs 才用于让工具参数或响应匹配给定 JSON Schema。同时，官方明确提醒：即使结构满足 Schema，模型仍可能把字段值填错，因此需要示例、拆分任务或额外验证。[OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)

**对 STKB 的推断。** STKB 当前 Pydantic `extra="forbid"` 和分类白名单是必要的，但它们只回答“能不能解析”和“枚举是否合规”。下一版合同应把 `content` 从自由字典变成按对象类型区分的联合结构，例如：

```json
{
  "objectType": "CUSTOMER_OBJECTION",
  "content": {
    "objectionExpression": "客户对价格/保障范围的原始表达",
    "customerConcern": "客户真正担忧的事项",
    "triggerContext": "在什么场景、阶段或问法下出现",
    "handlingDirection": "应对方向或判断原则",
    "recommendedResponse": "有原文依据的完整应对表达",
    "constraints": ["不可承诺或必须澄清的边界"],
    "applicableScope": {"product": "产品/版本", "audience": "客群"}
  }
}
```

这里的字段是 STKB 的设计建议，不是外部系统规定的字段。不同对象类型应有不同的必填字段：

| 对象类型示例 | 不应只保留的 `summary` | 建议至少验证的内容维度 |
| --- | --- | --- |
| `PRODUCT_FACT` | 一句话产品介绍 | 事实主体、数值/条件、产品或版本范围、生效/失效信息、限制或排除项、证据 |
| `BUSINESS_PROCESS` | “介绍购买流程” | 参与角色、步骤顺序、进入条件、分支/例外、输出或下一步、适用渠道、证据 |
| `CUSTOMER_OBJECTION` | “客户担心价格” | 原始异议、潜在顾虑、触发上下文、处理方向、允许话术、禁用承诺、证据 |
| `DECISION_RULE` | “根据客户情况推荐” | 前置条件、判断变量、动作、结果、例外、优先级、引用事实、证据 |
| `FAQ` | 一个问题和简短答案 | 问法变体、答案、答案依赖的事实/流程、适用范围、不可回答条件、证据 |

这些字段的核心目的不是让对象“写得更长”，而是让它可以被独立审核、归并、更新和下游消费。字段不适用时应显式输出 `null` 或 `not_stated` 及原因，而不是由模型补写。

### 2.2 来源 grounding 的最低合同

**外部事实。** LangExtract 的官方 README 说明：抽取值会映射到来源文本的精确字符位置；如果模型从示例而不是输入文本中生成内容，无法在源文档中定位的结果会出现空的 `char_interval`，可据此过滤。官方还要求 few-shot 示例中的抽取文本尽量逐字来自示例文本并保持顺序，以便做 Prompt alignment 检查。[LangExtract README：grounding 与示例对齐](https://github.com/google/langextract/blob/main/README.md#quick-start)

**对 STKB 的推断。** 现有 `evidence: list[str]` 只能说明“引用了某个锚点”，不能说明该锚点中的哪句话支持哪个字段。下一版应至少把证据建模为：

```json
{
  "evidence": [
    {
      "anchorId": "DP-...#section-7",
      "quote": "原文逐字摘录，不改写",
      "start": 18320,
      "end": 18408,
      "supports": ["content.handlingDirection", "content.recommendedResponse"],
      "evidenceKind": "direct_statement"
    }
  ]
}
```

硬校验应包括：

- `quote` 必须能在本次输入的完整 Markdown 中定位；`start/end` 与字符编码规则一致。
- 片段必须属于允许的 `anchorId` 和当前文档分段；分段调用要保存“分段局部偏移 → 全文偏移”的映射。
- 每个事实性字段至少关联一条证据；推断性字段必须明确 `inferred` 并进入复核，不得写成事实。
- 证据范围不能无界地覆盖整个文档；过宽的证据应视为“定位失败”，要求重新抽取。
- 证据内容不等于对象正文；正式 Markdown 可以整理，但不能新增无证据事实。

### 2.3 长文档：分段不是独立知识识别

**外部事实。** LangExtract 通过分块、并行和多轮抽取提高长文档召回；GraphRAG 也以 text unit 为抽取单元，并在后续对实体、关系和主张做跨单元汇总。[LangExtract 长文档说明](https://github.com/google/langextract/blob/main/README.md#scaling-to-longer-documents) [GraphRAG 方法](https://microsoft.github.io/graphrag/index/methods/)

**对 STKB 的推断。** 当前分段并发可以保留为上下文控制手段，但需要增加一个“跨段对象化”步骤：

1. 每个分段只负责高召回地产出带精确证据的原子主张、实体提及和关系提及。
2. 以全文范围的稳定 `claim_id` 或证据指纹汇总重复主张，保留来源列表和冲突。
3. 再由模型结合模块规则把主张聚合成对象；对象聚合必须输出拆分/合并理由和覆盖的主张 ID。
4. 程序做字段完整性、证据可定位、模块闭集、关系端点和重复身份校验。

若文档足够短，仍可进行一次全文识别；但无论调用几次，最终对象都必须能回溯到全文证据，而不是把分段结果列表直接当作正式对象。

## 三、实体、关系和属性图抽取

### 3.1 GraphRAG 的可借鉴部分

**外部事实。** Microsoft GraphRAG 的标准流程以 text unit 为输入，使用 LLM 抽取实体、关系和可选 claims；随后汇总同一实体/关系在多个 text unit 中的描述，再构建社区和报告。其输出明确区分 documents、text_units、entities、relationships、covariates 和 community_reports，并保留 text unit ID 作为回溯线索。[GraphRAG 输出结构](https://microsoft.github.io/graphrag/index/outputs/) [GraphRAG 方法](https://microsoft.github.io/graphrag/index/methods/)

GraphRAG 同时提供 Standard 和 FastGraphRAG。官方说明 Fast 方案使用 NLP noun phrase 和共现关系来降低成本，但图更嘈杂、对外部图探索的直接适用性更弱；如果重视高保真实体和图探索，建议保留标准方式。[GraphRAG 方法选择](https://microsoft.github.io/graphrag/index/methods/)

**对 STKB 的推断。** STKB 可以借鉴“中间账本”而不是照抄 GraphRAG 的业务对象：

```text
DocumentPackage
  └─ TextUnit / SourceAnchor
      ├─ AtomicClaim（原子主张）
      ├─ EntityMention（实体提及）
      └─ RelationMention（关系提及）
             ↓ 规则与对象化
       Candidate Knowledge Object
             ↓ 归一/归并/审核
       KnowledgeObject + 正式关系
```

建议每条实体或关系提及保存：

- 原文名称及精确证据区间；
- 建议类型和规范化候选，但不在抽取时直接生成正式 ID；
- 关系类型、方向、端点和证据；
- `asserted` / `inferred` / `ambiguous` 状态；
- 来源 DocumentPackage、规则版本和运行 ID。

GraphRAG 的 community report 适合做全局检索摘要，不适合作为 STKB 的一个销售知识对象，因为它的身份来自图社区和运行过程，不是来自 STKB 对象的可独立更新边界。

### 3.2 LlamaIndex PropertyGraph 的可借鉴部分

**外部事实。** LlamaIndex 文档说明，`PropertyGraphIndex` 会对每个 chunk 应用一个或多个 `kg_extractors`，把实体和关系作为节点元数据再写入图；`SchemaLLMPathExtractor` 支持允许的实体类型、关系类型和实体间连接规则，并通过 Pydantic、结构化输出和逐路径验证实现严格模式。[PropertyGraphIndex 构建](https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/#construction) [SchemaLLMPathExtractor](https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/#schemallmpathextractor)

**对 STKB 的推断。** STKB 的关系规则应从“关系有无”升级为“关系合同”：

```text
关系类型
  + 合法源对象/实体类型
  + 合法目标对象/实体类型
  + 方向
  + 最低证据
  + 是否允许跨文档
  + 冲突/多值规则
```

例如 `ABOUT_PRODUCT` 可以要求 `KnowledgeObject → PRODUCT`，`APPLIES_TO_SCENARIO` 可以要求 `KnowledgeObject → SCENARIO`；不符合端点规则的关系直接进入拒绝/未决，不先写入 Neo4j。

当前阶段建议不把 LlamaIndex 作为生产依赖。可以先在独立实验脚本中验证其 Schema extractor 思路，最终仍由 STKB 的 Pydantic 合同、PostgreSQL 登记和 Neo4j 投影负责真实身份与版本。

### 3.3 STKB 的图边界

项目现有方案已把正式 Markdown 作为知识对象正文，把 PostgreSQL 作为身份、版本和关系登记，把 pgvector/Neo4j 作为派生投影。这个边界必须保持：

- 图里的节点和边不是抽取结果的事实源；
- 正式对象、业务实体和正式关系通过审核后再投影；
- 任何图边都应能回到对象修订、DocumentPackage、source anchor 和证据片段；
- 图谱不能反向创建一个没有证据、没有规则归属或没有正式对象身份的 KnowledgeObject；
- GraphRAG 的 summary、community 和 text unit 可以作为研究对照产物，不能混入正式销售知识目录。

## 四、质量评估方法

### 4.1 为什么不能只看 precision/recall

**外部事实。** GenRES 指出，生成式关系抽取的输出可能在语义上正确但与人工参考关系的字面形式不同，传统精确匹配 precision/recall 会低估质量；同时，固定关系/实体集合的 Prompt 也可能诱发幻觉。论文建议从来源文本验证 precision，并用软匹配评估 recall，同时关注主题相似性、唯一性、粒度、事实性和完整性。[GenRES 摘要与方法](https://arxiv.org/abs/2402.10744)

**对 STKB 的推断。** STKB 应同时保留可计算的硬指标和业务评审指标。任何单一“综合分”都不能替代分项结果；尤其不能用“候选很多”“覆盖了更多模块”证明效果好。

### 4.2 推荐指标

| 质量层 | 指标 | 定义/计算 | 用途与门槛建议 |
| --- | --- | --- | --- |
| 结构 | Schema validity rate | 解析成功且符合 JSON Schema/Pydantic 的响应数 ÷ 响应总数 | 程序硬门槛；正式写入前目标为 100%，失败响应必须保留原始输出和错误 |
| 证据定位 | Evidence locator rate | 所有证据中的 `quote/start/end/anchor` 能在输入全文中唯一定位的比例 | 正式写入硬门槛；目标 100% |
| 证据支持 | Evidence precision | 被人工或受控 entailment 判断为确实支持对应字段的证据数 ÷ 被抽取证据数 | 正式对象前建议先达到试点目标 ≥ 0.95；目标需用样本校准 |
| 证据覆盖 | Evidence recall | gold 对象所需的关键证据片段被结果覆盖的比例 | 识别不能只给“一个相关段落”；初期建议 ≥ 0.80，再按对象类型细化 |
| 模块归属 | Macro-F1 / confusion matrix | 以 gold 对象的主模块/对象类型为标签，逐模块计算 P/R/F1，再做 macro 平均 | 防止高频 D1 模块掩盖 D3/D4 漏识别；同时保留 unresolved/none 类 |
| 对象召回 | Object recall | gold 对象中被结果匹配到的对象数 ÷ gold 对象数 | 匹配依据为主体、范围、类型和核心事实的有界软匹配，不要求标题字面相同 |
| 对象精度 | Object precision | 被匹配为 gold 对象的输出对象数 ÷ 输出对象数 | 发现过度抽取、把普通说明误当知识或重复生成 |
| 内容完整性 | Required-field coverage | 每类对象已填且有证据支撑的必填字段数 ÷ 该类对象必填字段数 | 用于回答“几十字 summary 是否够”；字段缺失不得仅靠摘要补偿 |
| 粒度 | Over-split rate | gold 的一项对象被拆成两个或以上输出对象的比例 | 发现按句子/字段/模块机械拆分；建议试点 ≤ 0.10 |
| 粒度 | Under-split rate | 两项或以上独立 gold 对象被合成一个输出对象的比例 | 发现把不同产品/版本/适用范围混在一起；建议试点 ≤ 0.15 |
| 唯一性 | Redundancy / uniqueness | 输出对象之间在规范化主体、范围、类型和核心事实上的重复比例 | 与 GenRES 的 uniqueness 对照，避免数量膨胀 |
| 稳定性 | Rerun object-set agreement | 同一输入、同一规则、同一模型配置重复运行时，对象匹配集合的 Jaccard/F1 | 观测随机性和身份不稳；初始试点目标 ≥ 0.90，需记录模型实际 determinism |
| 归并 | Merge action accuracy | `created/updated/reused/unchanged` 与人工 adjudication 一致的比例 | 验证跨资料知识对象是否被正确复用，而非重复创建 |
| 实体 | Entity mention / resolution P/R/F1 | 先评估实体提及，再评估是否归一到正确业务实体 | 将“识别到名字”和“确定它是谁”分开，不以字符串相等替代归一 |
| 关系 | Relation P/R/F1 + evidence | 关系类型、方向、端点和证据四者同时正确才计为严格正确 | 防止只看关系标签而忽略端点和来源 |

精确率、召回率和 F1 的基础定义可参考 [scikit-learn 官方指标文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)；F1 是 precision 与 recall 的调和平均。但上述对象匹配、证据支持和粒度指标是针对 STKB 的设计推断，不是 scikit-learn 或 GenRES 的现成接口。

### 4.3 对象粒度的专门验收

**外部事实。** GenRES 把 uniqueness 和 granularity 作为独立维度，并在其附录中用示例解释哪些关系可拆、哪些不应继续拆；DocRED 的文档级任务则说明跨句信息需要综合阅读，不能把单句结果当作全部事实。[GenRES 粒度说明](https://arxiv.org/abs/2402.10744) [DocRED](https://aclanthology.org/P19-1074/)

**对 STKB 的推断。** 业务对象的 gold 不应只标“有哪些句子”，还要标“哪些原子事实属于同一可更新、可复用对象”。对每份材料建立以下两层标注：

1. `AtomicClaim` 层：原文中可验证的产品事实、条件、动作、问答、异议、案例事件或评估规则。
2. `KnowledgeObject` 层：哪些 claim 共享业务主体、适用范围、生命周期、更新责任和消费方，应聚合到一项正式对象。

将预测对象和 gold 对象转成 claim 集合后，可计算：

- claim 对两两共聚的 pairwise precision/recall/F1；
- B³/CEAF 等聚类一致性指标作为辅助；
- over-split、under-split、冗余对象和跨版本错误合并的人工计数；
- 每个拆分或合并结果必须有对象边界说明，说明为什么共同更新或为什么独立更新。

不能规定“每 3500 字一个对象”“每个模块至少一个对象”或“每份资料必须十个对象”。

### 4.4 证据归因的专门验收

**外部事实。** Evidence Attribution 论文提出“citation masking and recovery”：遮住生成结果中的证据引用，再让标注者判断可恢复的证据；研究同时发现，最好模型也并非总能正确归因，人工精选证据优于机器选择证据。[NAACL 2025 论文](https://aclanthology.org/2025.naacl-long.282/)

**对 STKB 的推断。** 可以为每个候选对象做三项测试：

- **定位测试**：证据摘录是否逐字存在且位于正确原文位置。
- **支持测试**：只看该证据，业务 reviewer 是否能判断对象字段成立；若不能，标记 `evidence_insufficient`。
- **遮蔽恢复测试**：隐藏对象字段或证据 ID，只给对象主张和全文片段，reviewer 是否能找回同一证据。找不回时说明证据过宽、过弱或对象描述超出了原文。

LLM judge 可以批量发现疑似问题，但最终正式写入门槛仍应由程序定位校验 + 人工/抽样业务 adjudication 组成。

## 五、测试集和标注方案

### 5.1 建议的数据集形态

不要从模型已经抽出的 10 个对象倒推 gold。应以项目内原始资料为主建立固定样本集，初版建议：

- 12～20 份经过允许长期复现的脱敏 `DocumentPackage`；当前已有的产品培训、产品话术、合规说明、异议处理、分类规则可作为起点，但不能把当前模型结果直接当 gold。
- 按“资料类型”切分 train/dev/regression/holdout，不能只把同一材料随机切段后分到不同集合，避免原文泄漏。
- 至少 20% 样本由两名标注者独立标注，再由业务 adjudicator 处理冲突；记录标注分歧本身，因为它会暴露规则边界不清。
- 测试集同时包含正例、近似负例和应当输出 `unresolved` 的例子。

每份样本的 gold 至少包括：

```text
DocumentPackage / source anchor
  ├─ AtomicClaim：原子事实、条件、动作、表达及精确证据
  ├─ KnowledgeObject cluster：哪些 claim 应归为同一对象
  ├─ domain / module / object_type
  ├─ type-specific content fields
  ├─ BusinessEntity type + canonical candidate
  ├─ Relation type / direction / endpoints / evidence
  ├─ negative / unresolved reason
  └─ review note：为什么拆分、合并、拒绝或不确定
```

### 5.2 必测困难样本

| 场景 | 需要观察的失败 |
| --- | --- |
| 同一产品事实在多页重复出现 | 重复对象、证据合并、版本判断 |
| 产品版本或套餐不同 | 错误合并，尤其是身份线索只写“某产品” |
| 表格中的条件、金额、范围和脚注 | 丢失列头、单位、排除条件，产生不完整事实 |
| 话术与事实混排 | 把表达方式当成产品真值，或把事实复制成多个话术对象 |
| 客户异议、处理方向和完整话术并列 | D2/D3/D4 漂移，或者只抽出一句摘要 |
| 同一问答的多种问法 | 过度拆分 FAQ，忽略问法变体 |
| 跨页/跨段关系 | 分段后关系断裂，实体提及无法归一 |
| 合规/限制/禁用承诺 | 漏掉负向条件，形成危险的正向摘要 |
| 与销售知识无关的行政或系统描述 | 把运行规则、会话状态等误抽成销售知识 |
| 材料只出现弱线索 | 是否正确进入 `weak_signal` 或 `unresolved`，而不是凑模块覆盖 |

“不保存某次会话中的实时情绪、临时状态或 AI 扮演任务配置”这类内容属于系统/运行边界时，不应被混入销售知识模块规则或对象正文。它应放在系统治理/数据处理说明中，或在输入分类阶段被标记为非销售知识。这是对 STKB 当前页面文案和规则展示混淆风险的推断，不是外部资料结论。

### 5.3 基线与迭代方法

**外部事实。** OpenAI 的 Evaluation Flywheel 建议先分析失败样例，再对失败模式做结构化标注和基线测量，随后进行定向改进；官方特别强调，自动指标不能解释失败原因，人工检查和标注是产生改进路线的基础。[官方 Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/Building_resilient_prompts_using_an_evaluation_flywheel.md)

**对 STKB 的推断。** 下一轮实验应至少固定以下变量：

1. 同一组原始资料、同一组 gold 和同一模型版本。
2. 一次只改变一个主要变量：规则边界、对象 Schema、Prompt 示例、分段策略或模型参数。
3. 每次运行记录规则版本/指纹、Prompt 版本、Schema 版本、模型、温度、分段、调用次数、原始输出和质量指标。
4. 先看总体指标，再看 D3.1/D3.3/D4.1/D4.2 等高风险混淆矩阵和对象级错误样例。
5. 任何新规则必须在 holdout 和旧回归集上共同通过，不能只修复一个样本截图。

建议第一轮基线对比：

```text
Baseline A：当前 v0.4 规则 + 自由 content.summary + 一阶段对象识别
Baseline B：类型化 content + 精确证据 + 规则边界/负例 + 两阶段主张→对象
```

两者必须使用同一资料、同一模型和同一评估集。只比较候选数或页面观感没有意义。

## 六、对 STKB 的目标合同建议

### 6.1 规则包应从“说明文字”升级为可执行定义

当前 TOML 规则包已经有域、模块、对象类型、边界、来源和消费方，但下一版至少还需要：

- 模块的纳入条件和明确排除条件；
- 每个对象类型的 type-specific content Schema；
- 必填字段、可选字段和 `not_stated` 规则；
- 对象身份维度：主体、产品/版本、客群、场景、渠道、时间和生命周期；
- 拆分/合并例子以及反例；
- 与相邻模块的冲突裁决；
- 允许引用的业务实体类型和引用角色；
- 允许建立的关系类型、端点、方向和证据要求；
- 何时必须输出 `weak_signal` / `unresolved`；
- 合规、隐私和不应进入销售知识的内容边界。

规则包应继续作为版本化事实源，并由同一份定义生成：模型 Prompt、Pydantic/JSON Schema、程序校验和 Web 规则审阅数据。规则页只能展示当前运行实际使用的版本，不能再有与运行无关的“计划建设”等标签。

### 6.2 推荐的模型调用分层

建议把一次 Web 操作编排成多阶段，但不要求使用者手工操作多次：

```text
阶段 A：资料范围与证据发现
  输入：全文 Markdown / 结构分段 + 规则中的销售知识边界
  输出：AtomicClaim、EntityMention、RelationMention、精确证据

阶段 B：对象化与分类
  输入：主张集合 + 完整模块规则 + 类型化对象 Schema
  输出：0～N 个候选对象、对象覆盖 claim IDs、模块归属、拆分/聚合理由

阶段 C：程序硬校验
  校验：Schema、枚举、字段完整性、证据定位、模块边界、关系端点、重复

阶段 D：归一与归并
  输入：通过校验的候选 + 已有实体/对象的有限候选集
  输出：复用/新增/更新/冲突，不让相似度单独决定身份

阶段 E：正式化
  条件：对象内容完整、证据充分、身份稳定、无未解决冲突
  输出：PostgreSQL 登记 + 一对象一份 Markdown；之后才投影 pgvector/Neo4j
```

这个流程仍然可以由一个按钮启动，但 Web 端要显示每个阶段的真实输入、输出、错误和计数。模型不需要输出隐藏思维过程；需要输出的是可审计字段、证据、状态和理由。

### 6.3 正式对象写入门槛

以下是 STKB 的建议硬门槛，不是外部项目的承诺：

- 结构化输出可解析且符合当前 Schema；
- 所有必填字段有值或明确 `not_stated`；
- 每个事实性字段至少有一个可定位证据；
- 模块、对象类型和关系端点均通过规则校验；
- 对象边界能够说明共同更新/复用原因；
- 归并候选已处理冲突，没有把两个产品/版本错误合并；
- 不存在未解决的高风险合规或身份问题；
- 正式 Markdown、PostgreSQL 登记和对象指纹一致。

没有通过门槛时，结果应该是“待复核/拒绝/未决”，不能为了让页面显示正式对象而写入。

## 七、实施优先级

### P0：先让质量可测、错误可回溯

1. 把 `content` 改为按对象类型的强 Schema，去除只靠 `summary` 即可通过的路径。
2. 把 `evidence` 改为精确摘录 + 字符区间 + 支持字段，并维护分段到全文的偏移映射。
3. 建立 12～20 份样本和最小 gold，先标注对象边界、证据和模块归属。
4. 在正式写入前增加结构、证据、字段完整性和高风险冲突硬门槛。
5. Web 调试页优先显示对象质量报告：缺失字段、无证据字段、过度拆分、疑似重复、未决原因；不要只显示对象数。

### P1：改善识别质量和对象粒度

1. 将证据/原子主张发现与对象化分类拆成两阶段调用，由一次 Web 操作编排。
2. 加入模块正反例、相邻模块裁决例和“不属于销售知识”例子；few-shot 证据必须逐字对齐原文。
3. 对同一资料重复运行，测对象集合稳定性；对跨资料运行，测 created/updated/reused 的一致性。
4. 用错误分类台账驱动 Prompt 和规则迭代，而不是凭单次模型输出直接改规则。

### P2：关系和投影验证

1. 在对象质量达到门槛后，增加实体归一和关系合同验证。
2. 用 GraphRAG/LlamaIndex 做独立对照实验，验证中间账本、Schema 路径约束和图检索的收益。
3. 只将正式对象和正式关系投影到 pgvector/Neo4j，保留对象修订与证据回链。

### P3：持续回归和模型选择

1. 把样本、gold、grader、错误标签和运行轨迹纳入可复现评估目录。
2. 对模型、Prompt、规则和分段策略做矩阵对比，发布每轮结果与成本/时延。
3. 保留人工 adjudication 作为新错误的发现机制；LLM judge 只做批量预筛和排序。

## 八、事实与推断边界

| 内容 | 类型 | 依据 |
| --- | --- | --- |
| LangExtract 支持精确字符区间、分段并行、多轮抽取和可视化 | 外部事实 | [Google LangExtract README](https://github.com/google/langextract/blob/main/README.md) |
| OpenAI Structured Outputs 可用严格 JSON Schema 约束形状，但不能保证字段语义正确 | 外部事实 | [OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) |
| GraphRAG 有 text unit、实体、关系、claims、社区报告等中间/派生产物 | 外部事实 | [GraphRAG outputs](https://microsoft.github.io/graphrag/index/outputs/) |
| LlamaIndex SchemaLLMPathExtractor 支持实体/关系闭集、端点约束和严格校验 | 外部事实 | [LlamaIndex Property Graph 文档](https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/#schemallmpathextractor) |
| 生成式抽取需要测唯一性、粒度、事实性和完整性 | 外部事实 | [GenRES](https://arxiv.org/abs/2402.10744) |
| 文档级关系需要跨句证据，推荐—修订式标注可能有 false negative | 外部事实 | [DocRED](https://aclanthology.org/P19-1074/)、[Revisiting DocRED](https://aclanthology.org/2022.emnlp-main.580/) |
| 现有 STKB `content` 为自由 dict、证据只有锚点 ID、归并使用 identity hints hash | 仓库事实 | [models.py](../../services/app/features/sales_knowledge_identification/models.py)、[service.py](../../services/app/features/sales_knowledge_identification/service.py)、[formalizer.py](../../services/app/features/sales_knowledge_identification/formalizer.py) |
| STKB 应采用“证据/主张→对象化→硬校验→归一归并→正式化”分层 | 对 STKB 的推断 | 综合当前代码问题与上述外部机制，不是任何一个外部项目的现成架构 |
| 对象数不能预设，摘要长度不能作为完整性标准 | 对 STKB 的推断 | 基于对象身份/边界/字段合同和 GenRES 粒度、完整性维度 |
| GraphRAG community report 不应直接成为 STKB KnowledgeObject | 对 STKB 的推断 | 基于 GraphRAG 派生产物语义与 STKB 正式对象事实源边界 |

## 九、研究来源清单（直接链接）

1. Google，**LangExtract** 官方仓库与 README：<https://github.com/google/langextract>
2. OpenAI，**Introducing Structured Outputs in the API**：<https://openai.com/index/introducing-structured-outputs-in-the-api/>
3. JSON Schema，**Specification**：<https://json-schema.org/specification>
4. Microsoft，**GraphRAG 官方仓库**：<https://github.com/microsoft/graphrag>
5. Microsoft，**GraphRAG Indexing Overview**：<https://microsoft.github.io/graphrag/index/overview/>
6. Microsoft，**GraphRAG Methods**：<https://microsoft.github.io/graphrag/index/methods/>
7. Microsoft，**GraphRAG Outputs**：<https://microsoft.github.io/graphrag/index/outputs/>
8. LlamaIndex，**Using a Property Graph Index**：<https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/>
9. LlamaIndex，**SchemaLLMPathExtractor 源码**：<https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/property_graph/transformations/schema_llm.py>
10. OpenAI，**Evals 官方仓库**：<https://github.com/openai/evals>
11. OpenAI Cookbook，**Building resilient prompts using an evaluation flywheel**：<https://github.com/openai/openai-cookbook/blob/main/examples/evaluation/Building_resilient_prompts_using_an_evaluation_flywheel.md>
12. Jiang et al.，**GenRES: Rethinking Evaluation for Generative Relation Extraction in the Era of Large Language Models**：<https://arxiv.org/abs/2402.10744>
13. Yao et al.，**DocRED: A Large-Scale Document-Level Relation Extraction Dataset**：<https://aclanthology.org/P19-1074/>
14. Tan et al.，**Revisiting DocRED - Addressing the False Negative Problem in Relation Extraction**：<https://aclanthology.org/2022.emnlp-main.580/>
15. Ma et al.，**DREEAM: Guiding Attention with Evidence for Improving Document-Level Relation Extraction**：<https://aclanthology.org/2023.eacl-main.145/>
16. Xing et al.，**Evaluating Evidence Attribution in Generated Fact Checking Explanations**：<https://aclanthology.org/2025.naacl-long.282/>
17. scikit-learn，**Precision / Recall / F-score API**：<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html>
