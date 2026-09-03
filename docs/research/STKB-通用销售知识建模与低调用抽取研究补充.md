# STKB 通用销售知识建模与低调用抽取研究补充

> 研究日期：2026-08-28
> 研究目的：补充上一份《STKB-销售知识对象抽取质量调研》，为通用销售知识域重审、模块/对象边界、低调用文档级抽取和可读正式文件提供一手依据。
> 研究范围：信息抽取与事件/关系建模、Schema-guided extraction、长文档分块与全局合并、结构化输出与约束解码、文档级抽取评估、通用销售流程建模。
> 证据口径：带“外部事实”的内容直接依据官方规范、第一方文档或论文原文；带“对 STKB 的推断/设计判断”的内容是结合项目目标得出的建议，不是外部系统已经证明的结论。未运行真实模型，未访问数据库。

## 结论先行

1. 本次检索没有发现一份被通用标准组织采纳、能够直接规定“跨行业销售知识模块和对象类型”的统一销售知识本体。可复用的共识主要来自三类标准：APQC 提供可定制的跨行业流程分类，BPMN 描述流程中的活动/事件/网关，DMN 描述决策与业务规则。它们是建模边界和语义层次的参考，不是 STKB 的 22 模块答案。
2. 信息抽取研究支持“一个结构化抽取任务同时产出实体、关系、事件或主张”的方向，但不支持把一个 JSON Schema 合法响应当成业务知识正确。Schema 只能约束形状；证据定位、字段语义、对象粒度、冲突和生命周期仍需独立验证。
3. 长文档研究的共同模式是“局部发现 + 全局整合”，而不是“每个分段直接生成正式对象”。分段输出必须保留原文证据、局部到全文的偏移和中间主张，之后才做对象化、归并和冲突处理。
4. 对 STKB 当前常见规模资料，2—4 次模型调用可以作为可测量的工程预算和实验假设：一次发现/结构化抽取、一次对象化与全局合并，必要时增加一次定向修复或一次长文档补充。它不是理论定理，也不适用于任意超长文档；超长文档的调用数应由输入 token、分段策略和质量失败触发条件决定。
5. 正式知识文件应采用“YAML front matter + 人可读正文 + 可选机器旁车文件”。结构化 JSON 可以保留在独立 `.json`/运行证据中，或作为折叠附录，但不能继续把整个正文写成 JSON fenced block。

## 一、来源与可复用边界

| 来源 | 一手证据 | 对 STKB 的具体影响 | 边界/不可照搬 |
| --- | --- | --- | --- |
| [UIE：Unified Structure Generation for Universal Information Extraction](https://aclanthology.org/2022.acl-long.395/) | 论文用统一结构语言和 schema-based prompt 处理实体、关系、事件、情感等不同 IE 任务。 | 规则包可以把“抽取结构”和“销售模块/业务对象”分层；一次抽取任务可发现多种中间结构，不必按模块拆成多路调用。 | 论文是训练模型框架，不证明任意通用 LLM 在 STKB 销售资料上都能一次正确抽取，也不提供 STKB 的字段定义。 |
| [OneIE：A Joint Neural Model for Information Extraction with Global Features](https://aclanthology.org/2020.acl-main.713/) | 联合抽取实体、事件触发词及其链接，并在解码中利用跨子任务和跨实例的全局特征。 | 支持在中间层保留 `claim/entity/relation/event` 之间的连接，而不是分别抽取后丢失关联。 | OneIE 的实验主要是句子级 IE；不能把其结果直接外推为长文档销售知识正确性。 |
| [Document-level Event Extraction with Efficient End-to-end Learning of Cross-event Dependencies](https://aclanthology.org/2021.nuse-1.4/) | 文档级事件的论元可能散落在不同句子，跨句共指和事件依赖是核心困难。 | 对象化前必须允许跨段/跨页证据合并；“分段结果列表”不能直接作为正式对象。 | 该论文针对事件抽取，不等于所有销售知识都应建成事件。 |
| [DocEE](https://aclanthology.org/2022.naacl-main.291/) / [DocEE-zh](https://aclanthology.org/2024.findings-emnlp.35/) | 文档级事件数据集采用人工标注和细粒度论元；DocEE-zh 说明中文文档级论元抽取仍然困难。 | STKB Gold 应标注原子主张、论元/字段、跨句证据和对象聚合，而不是只标对象标题或模块命中。 | 数据集来自新闻/事件领域；不能把事件类型移植成销售模块。 |
| [OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) / [JSON Schema 规范](https://json-schema.org/specification) | 严格结构化输出可约束响应匹配给定 JSON Schema；JSON Schema 的 Core 与 Validation 定义结构和校验语义。OpenAI 同时明确说明，Schema 合法不排除字段值错误，并存在 Schema 子集、拒答、截断和首次 Schema 处理延迟等边界。 | 将 Schema validity 作为第一层硬门槛；另设证据支持、语义正确性、对象边界和安全/合规审查。规则定义需区分校验关键词、说明性元数据和业务判断。 | 不假设当前 OpenAI-compatible gateway 支持同等严格解码；不把 `required`、`enum`、最小长度当作业务真值。 |
| [PICARD](https://aclanthology.org/2021.emnlp-main.779/) | 通过增量解析在每个解码步骤拒绝不合法 token，减少形式语言输出无效的情况。 | 可把约束解码理解为语法层防线，适用于 JSON/语法合法性；模型参与步骤仍要保留原始输出、拒答和截断证据。 | PICARD 以 SQL 形式语言为实验对象；它不能判断“原文是否支持这个销售事实”。 |
| [Google LangExtract README](https://github.com/google/langextract/blob/main/README.md) / [长文档示例](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md) | 官方库将抽取值对齐到原文字符区间，并用分块、并行、多轮提高长文档召回；文档说明多轮结果需要合并。 | 采用 `quote + start/end + source_anchor + supports` 的字段级证据；分段只是上下文控制，不是对象边界。 | 官方长文档示例使用多 worker、多 pass，证明的是质量/吞吐取舍，不证明低调用次数。 |
| [Microsoft GraphRAG 方法](https://microsoft.github.io/graphrag/index/methods/) / [数据流](https://microsoft.github.io/graphrag/index/default_dataflow/) / [输出结构](https://microsoft.github.io/graphrag/index/outputs/) | 以 text unit 抽取实体、关系、claims，再跨 text unit 合并同一实体/关系并生成摘要；Fast 方案用更多传统 NLP 以降低成本，但图更嘈杂。 | 建立“文档—TextUnit—原子主张—候选对象”的中间账本，并把局部抽取与全局归并分开计量。 | GraphRAG 的实体/社区摘要是图检索产物，不是 STKB 正式 `KnowledgeObject` 的身份或事实源。 |
| [W3C PROV-DM](https://www.w3.org/TR/prov-dm/Overview.html) | 用 domain-agnostic 的 Entity、Activity、Agent 及使用、生成、派生、归属关系表达数据来源和处理过程。 | 运行证据可按“输入文档/分段/Prompt/模型响应/候选对象”为实体，“一次模型调用/校验/正式化”为活动建立可追溯链。 | 不要求 STKB 直接实现完整 PROV-O；这是语义检查参考，项目仍可用现有运行记录字段实现。 |
| [OMG BPMN 2.0.2](https://www.omg.org/spec/BPMN/2.0.2/PDF/) / [OMG DMN 1.4](https://www.omg.org/spec/DMN/1.4/PDF) | BPMN 将流程表示为活动、事件、网关和顺序流；DMN 区分输入数据、决策、业务知识模型和知识来源，并设计为可与 BPMN 协同。 | 用流程、决策/规则、知识来源三个层次审查相邻模块：流程描述“如何推进”，规则描述“依据什么条件决定”，知识来源描述“谁/什么授权或支撑”。 | 标准提供建模语言而非销售知识字段；不应把 BPMN/DMN 直接当作 STKB 的产品 Schema。 |
| [APQC Cross-Industry PCF](https://www.apqc.org/resource-library/resource-listing/apqc-process-classification-framework-pcf-cross-industry-pdf-7) | PCF 提供跨行业流程分类和共同语言，但明确它是可定制框架，并非每个组织都拥有其中全部流程；另有行业特化版本。 | 可用“跨行业流程骨架 + 组织/行业扩展”检验模块是否混入保险特定概念；不要把一份行业资料变成通用模块边界。 | PCF 是流程分类和基准框架，不是销售知识对象本体，也不替代字段级证据。 |
| [Microsoft Dynamics 销售流程概览](https://learn.microsoft.com/en-us/dynamics365/project-operations/sales/sales-overview) / [Salesforce Sales Stages](https://help.salesforce.com/s/articleView?id=sf.essentials_sales_stages.htm&language=en_US) | Microsoft 明确产品型与项目型销售周期可不同，流程阶段可配置、条件化；Salesforce 说明销售阶段应按企业业务自定义。 | “阶段”必须带组织/流程/场景作用域和生命周期，不应被硬编码为跨行业固定序列；这也支持把行业差异放在扩展层。 | 两者是厂商产品实践，不是跨行业标准；只能作为差异性证据，不能作为 STKB 模块清单。 |

## 二、通用销售知识模型的语义边界

### 2.1 没有证据支持固定的跨行业 22 模块

**外部事实。** APQC 的跨行业 PCF 是用于共同命名、组织和比较流程的可定制框架，并明确不同组织不一定拥有其中所有流程；APQC 同时维护行业特化版本。[APQC PCF](https://www.apqc.org/resource-library/resource-listing/apqc-process-classification-framework-pcf-cross-industry-pdf-7) Microsoft 的销售流程文档则说明，项目型和产品型销售周期不同，阶段可以按条件配置。[Microsoft 销售流程概览](https://learn.microsoft.com/en-us/dynamics365/project-operations/sales/sales-overview) Salesforce 也将销售阶段定位为企业可自定义的流程表达。[Salesforce Sales Stages](https://help.salesforce.com/s/articleView?id=sf.essentials_sales_stages.htm&language=en_US)

**对 STKB 的推断。** 本次检索没有找到可直接作为“通用销售知识 5 域 22 模块”的公认一手标准；这是一条检索范围内的结论，不是绝对证明“世界上不存在任何销售本体”。因此重审时应从消费任务和对象生命周期反推模块，不从保险材料中倒推模块数量。D1—D5 是否保留、合并或重命名应由以下证据决定：

- 模块是否对应不同的业务问题和消费动作；
- 模块中的对象是否拥有独立的身份、更新触发和证据要求；
- 相邻模块是否会由同一消费者在同一时刻同时读取，且字段无法判别来源；
- 跨行业样例是否能出现该语义，而不是只在保险资料中出现；
- 模块能否通过对象边界和消费接口被测试，而不是只靠名称、枚举或最小字符数存在。

### 2.2 BPMN/DMN 为拆开“流程、规则、策略”提供了语义参照

**外部事实。** BPMN 的过程模型由活动、事件、网关和顺序流构成，强调工作如何从触发走向结果。[BPMN 2.0.2](https://www.omg.org/spec/BPMN/2.0.2/PDF/) DMN 将 Decision 定义为由输入和决策逻辑确定输出，将 Business Knowledge Model 作为封装业务规则、决策表或分析模型的知识单元，并将 Knowledge Source 单列为权威来源。[DMN 1.4](https://www.omg.org/spec/DMN/1.4/PDF)

**对 STKB 的设计判断。** 这不要求 STKB 直接采用 BPMN/DMN 文件格式，但可用它们作为模块审查的“语义试纸”：

| STKB 候选语义 | 应回答的问题 | 不应混入 |
| --- | --- | --- |
| 流程/阶段知识 | 由什么触发？角色按什么顺序做哪些动作？有哪些分支、例外、输出和下一步？ | 某个客户条件下应该选什么方案的决策逻辑 |
| 决策/规则知识 | 输入变量是什么？判断条件和优先级是什么？输出动作/结果是什么？何时不能应用？ | 把具体执行步骤、整段话术或产品事实复制进规则正文 |
| 策略/打法知识 | 面向什么目标和情境？选择哪种方向？需要观察什么信号？如何调整？ | 把确定性门槛冒充策略，或把策略建议写成产品事实 |
| 表达/话术知识 | 面向谁、在什么上下文说什么？哪些表达是推荐、可替换、禁用？ | 把话术本身当成事实或规则的唯一证据 |
| 来源/治理知识 | 谁授权、何时生效、适用范围和审阅状态是什么？ | 把运行 trace、临时会话状态写成销售知识 |

这张表是 STKB 的设计判断，不是 BPMN/DMN 对销售领域的规定。它的直接用途是审查潜在重叠，例如“政策流程”与“合规规则”是否分别属于流程与决策约束，“触发机制”与“策略决策”是否分别属于可观察信号与决策输出，而不是预先宣布哪一项必须保留。

### 2.3 阶段不是通用事实，应是有作用域的属性或对象

**外部事实。** Microsoft 的流程示例包含 Qualify、Estimate、Internal review、Contract、Deliver、Close 等阶段，但同时说明项目型流程可有更多估算和审查阶段，且某些阶段只在满足条件时出现。[Microsoft 销售流程概览](https://learn.microsoft.com/en-us/dynamics365/project-operations/sales/sales-overview) Salesforce 允许企业创建自定义销售阶段。[Salesforce Sales Stages](https://help.salesforce.com/s/articleView?id=sf.essentials_sales_stages.htm&language=en_US)

**对 STKB 的推断。** 不宜在通用核心中固定“线索—需求—报价—签约”的唯一阶段树。可以保留一个跨行业的 `stage` 语义，但必须至少绑定：`process_scope`、`organization_or_program`、`stage_order`（可选）、`entry_condition`、`exit_evidence`、`exceptions` 和生效范围。行业扩展可以增加“核保/理赔”等词汇，但它们不能进入通用核心字段的定义。

## 三、文档级抽取与低调用架构

### 3.1 研究共同支持的抽取分层

**外部事实。** UIE 说明不同 IE 任务可以统一为结构化生成，并通过 schema prompt 指定要抽取的目标结构。[UIE](https://aclanthology.org/2022.acl-long.395/) OneIE 说明实体、事件触发词和链接之间存在跨子任务依赖，联合解码可以保留这些依赖。[OneIE](https://aclanthology.org/2020.acl-main.713/) 文档级事件研究和 DocEE 系列说明，论元常跨句分布，需要文档范围内的整合和细粒度人工标注。[文档级事件抽取](https://aclanthology.org/2021.nuse-1.4/) [DocEE](https://aclanthology.org/2022.naacl-main.291/) [DocEE-zh](https://aclanthology.org/2024.findings-emnlp.35/)

**对 STKB 的设计判断。** 推荐中间层按以下语义分开，而不是按 22 个模块拆成 22 条模型流水线：

```text
DocumentPackage / SourceAnchor
  → AtomicClaim + EntityMention + RelationMention + Event/ProcessMention
  → 规则约束下的 module/object_type 分类
  → KnowledgeObject 聚合、拆分、冲突与身份候选
  → 程序硬校验与待复核
  → 正式 Markdown / JSON sidecar / 投影
```

其中：

- `AtomicClaim` 保存一个可被原文支持的最小事实、条件、动作、表达或否定约束；它不是正式 KnowledgeObject。
- `EntityMention` 保存原文提及和候选规范名；正式业务实体 ID 应在后续归一阶段决定。
- `RelationMention`/`EventMention` 只表达原文出现的关系或事件结构；它们不能绕过对象身份直接创建正式图边。
- `KnowledgeObject` 的边界由主体、作用域、生命周期、更新责任和消费者共同决定，不由段落长度或模块数量决定。

### 3.2 分段与全局合并：证据优先于摘要

**外部事实。** LangExtract 用字符区间把抽取值对齐到原文，并使用按文本边界分块、并行和多轮抽取提高长文档召回；其长文本示例还说明多轮结果需要处理重叠和新增结果。[LangExtract README](https://github.com/google/langextract/blob/main/README.md) [长文档示例](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md) GraphRAG 的标准数据流先按 TextUnit 抽取实体、关系和 claims，再将相同实体/关系的局部描述合并后摘要；其输出保留 `text_unit_ids` 作为来源回链。[GraphRAG 数据流](https://microsoft.github.io/graphrag/index/default_dataflow/) [GraphRAG 输出](https://microsoft.github.io/graphrag/index/outputs/)

**对 STKB 的设计判断。** 分段调用的最低中间合同应包括：

```text
chunk_id
full_text_start / full_text_end
section_path / context_prefix
claim_id（全局稳定或可由证据指纹重建）
quote + local_start/end + full_start/end
claim_kind（fact / condition / action / expression / prohibition / unresolved）
```

全局合并必须至少处理：

1. 同一主张在多个片段重复出现时合并证据，而不是简单字符串去重；
2. 同一主体的不同版本、适用范围或时间条件冲突时保留冲突，不强行覆盖；
3. 只有共享更新边界和消费者的主张才聚合为同一对象；
4. 每个对象字段都能回链到一个或多个主张及其全文区间；
5. 不能在分段中看到的全局结论必须进入 `unresolved` 或待复核，而不是由摘要补写。

GraphRAG 的“实体/关系摘要”适合说明如何做全局汇总，但不是 STKB 的对象身份算法；LangExtract 的多 pass 适合解释召回—成本权衡，也不应被直接变成每个文档的默认多轮调用。

### 3.3 2—4 次调用的依据、公式与局限

这里的“调用”指一次实际模型请求/响应；并发不会把多个请求变成一次调用。对当前常见长度资料，建议先验证下列自适应编排：

```text
调用 1：全文可容纳时，发现 AtomicClaim/Entity/Relation/Event + 精确证据
        超过上下文阈值时，改为按语义边界的 chunk discovery（请求数随 chunk 数增长）
调用 2：使用压缩后的主张账本做对象化、模块归属、拆分/合并与身份候选
调用 3：只有硬校验失败或高风险字段证据不足时，做一次定向修复/补证据
调用 4：仅在修复后仍有单一可定位冲突时，做一次针对冲突的复核
程序步骤：规范化、证据定位、Schema 校验、重复/冲突检查、正式化不调用模型
```

可观测的次数公式为：

```text
N_calls = N_discovery_batches + 1_objectize + N_targeted_repair
N_discovery_batches = 1                         (全文输入不超阈值)
                     = ceil(source_tokens / chunk_capacity)  (超阈值)
N_targeted_repair ∈ {0, 1, 2}
```

**为什么有依据：** UIE/OneIE 说明不同抽取子任务可以联合建模，支持把模块级调用收敛到一个结构化发现任务；LangExtract/GraphRAG 说明长文档可采用“局部抽取—全局合并”的分层；OpenAI Structured Outputs 的官方限制说明，当 Schema 合法但字段值仍可能错误时，应提供示例或拆分成更简单的子任务。[OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)

**为什么不是定理：** 2—4 不是任何文档长度、模型、规则复杂度都保证足够的数字。若资料超过上下文窗口，`N_discovery_batches` 可能大于 1；若资料含大量表格、脚注、跨页条件或高风险否定，仍可能需要额外复核。当前药享保运行的 32 次调用是项目运行基线和成本风险信号，外部来源不能证明这 32 次“必要”或“错误”；必须用同一资料对比。

**建议的测量目标（设计判断，需先测当前基线）：**

| 指标 | 标准长度资料的首轮目标 | 长文档边界 |
| --- | --- | --- |
| 模型调用数 | `p95 ≤ 4`，禁止按对象类型或每个对象各发一次请求 | 按 `N_discovery_batches` 线性记录，不把并发掩盖成一次 |
| 结构成功率 | 解析并通过 Schema 的最终响应 100%；失败保留原始响应和错误 | 逐批统计，不能用最终对象数替代 |
| 证据定位率 | `quote/start/end/anchor` 全部可在全文定位；正式对象目标 100% | 每个 chunk 和全局对象分别统计 |
| 语义质量 | 与人工 adjudication 的对象精确率、召回率、模块 macro-F1 相对当前基线不下降超过预先约定的容差（建议先用 5 个百分点做试验门槛） | 单独按资料类型、跨页关系和高风险约束报告 |
| Token | `Σ(prompt_tokens + completion_tokens)` 记录到调用级；首轮目标为不超过当前 32 次基线的 60%，若达不到必须解释规则包重复和上下文开销 | 以每千源 token 成本、每个有效对象成本归一化 |
| 耗时 | 记录 p50/p95 wall time；首轮目标 p95 不高于当前基线的 1.25 倍，同时调用数显著下降 | 并发 chunk 与串行 global merge 分开报告 |
| 修复 | `N_targeted_repair ≤ 2`，且只针对失败字段/证据，不重复生成全部对象 | 超限直接进入待复核，不无限重试 |

60% 和 1.25 倍是工程试验的初始门槛，不是论文或厂商承诺；真实基线测完后允许调整，但必须在实验前冻结，避免为了单次 Gold 结果事后改门槛。

## 四、结构化输出、证据和运行轨迹

### 4.1 Schema 合法不等于知识正确

**外部事实。** OpenAI 介绍的 Structured Outputs 通过受约束解码让响应匹配开发者提供的 JSON Schema，并说明 Schema 会被编译为语法以限制下一 token；同一文档也说明首次使用新 Schema 有额外处理延迟，支持的是 JSON Schema 子集，拒答或截断时可能没有完整结构，且字段值仍可能错误。[OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) JSON Schema 官方规范则将 Core 与 Validation 分开，并说明某些 annotation 关键词不参与验证。[JSON Schema Specification](https://json-schema.org/specification) [JSON Schema Annotations](https://json-schema.org/understanding-json-schema/reference/annotations)

**对 STKB 的设计判断：**

- `SchemaValid` 只表示形状合法；`EvidenceLocated` 表示证据存在；`FieldSupported` 表示证据足以支持字段；`ObjectAccepted` 才表示可以进入正式化。
- 如果网关不支持原生严格 Schema，后端要显式记录“未启用约束解码”，用 Pydantic/JSON Schema 解析和一次定向修复兜底，不能在页面上写成“严格结构化输出”。
- Schema 编译/缓存造成的首次延迟、拒答和 `max_tokens` 截断应进入调用明细，而不是被归为模型“耗时异常”。
- 所有需要事实依据的字段都要有字段级证据；`required` 不能代替证据，`minLength` 不能代替完整性。

### 4.2 运行证据可借鉴 PROV 的三层语义

**外部事实。** W3C PROV-DM 将 Entity、Activity、Agent 及使用、生成、派生、责任关系作为跨领域的 provenance 核心，并允许应用增加领域特化属性。[PROV-DM](https://www.w3.org/TR/prov-dm/Overview.html) [PROV-O](https://www.w3.org/TR/2013/PR-prov-o-20130312/)

**对 STKB 的设计判断。** 运行证据至少分三层：

1. **阶段结果层**：发现了多少主张、候选对象、拒绝项、未决项，当前状态如何流转；
2. **调用明细层**：每次模型调用的阶段、目的、输入范围、格式化 Prompt、Schema、输出摘要/原文、模型、token、耗时、重试、错误和关联结果 ID；
3. **来源派生层**：`DocumentPackage → chunk/claim → candidate → KnowledgeObject revision → Markdown/sidecar` 的派生链。

Web 页面应先展示阶段结果和数量变化，点击模型节点后再展开调用明细；这会让“模型参与了哪里”可见，同时避免把几十条调用全部铺在首屏。不要展示模型隐藏思维过程，展示可审计的输入、输出、状态和证据即可。

## 五、文档级抽取评估：从字段到对象再到成本

### 5.1 Gold 应同时标注原子事实和对象边界

**外部事实。** DocEE/DocEE-zh 使用人工标注的文档级事件和细粒度论元，表明只测句子级触发词不能覆盖跨句论元；DREEAM 将证据句作为关系抽取的独立监督信号，说明“找证据”和“判关系”可以分别评估。[DocEE](https://aclanthology.org/2022.naacl-main.291/) [DocEE-zh](https://aclanthology.org/2024.findings-emnlp.35/) [DREEAM](https://aclanthology.org/2023.eacl-main.145/)

**对 STKB 的设计判断。** 新 Gold 至少应包含：

```text
source_anchor / full-text span
AtomicClaim：主张类型、原文摘录、否定/条件、证据区间
object_cluster：哪些 claim 共同构成一个可更新/可消费对象
module / object_type：主归属与可选次级关系
type-specific fields：字段值、字段证据、not_stated/unresolved
identity scope：主体、版本、客群、场景、渠道、时间、生效状态
negative / unresolved reason
review note：为什么合并、拆分、拒绝或暂不确定
```

### 5.2 建议保留的指标与新增的成本指标

上一份调研已经提出 Schema、证据定位、证据支持、模块 macro-F1、对象 precision/recall、字段覆盖、过拆/欠拆、冗余、稳定性和归并准确率。本补充建议保留这些指标，并加上文档级与资源指标：

| 指标 | 定义 | 目的 |
| --- | --- | --- |
| Cross-unit evidence recall | 需要跨 chunk 的 gold 证据链被完整覆盖的比例 | 检查分段是否切断上下文 |
| Claim-to-object coverage | 被正确对象覆盖的 gold claim 数 ÷ gold claim 总数 | 检查对象摘要是否吞掉细节 |
| Partition agreement | 预测对象与 gold 对象对 claim 的共聚关系一致性 | 直接测过拆、欠拆和错误合并 |
| Unsupported critical field rate | 高风险事实/限制字段中没有充分证据的比例 | 保障合规和事实安全 |
| Call count / source token | 模型调用数 ÷ 文档源 token | 比较不同分段策略的可扩展性 |
| Token per accepted object | 总输入/输出 token ÷ 通过门槛的正式对象数 | 防止“输出很多但有效对象少” |
| Stage latency | 每阶段 p50/p95，另列串行和并发等待 | 区分模型请求、合并、校验耗时 |

### 5.3 评估方法的边界

文档级关系/事件数据集的 F1 不能直接证明 STKB 的销售对象质量；反之，当前 Gold 10/10 也不能证明模块体系、调用架构或 Markdown 文件可用。外部研究的共同启示是：先固定人工审阅的对象边界和证据，再用自动指标定位退化，最后抽查错误样例。评估必须同时回答“抽到了什么”“为什么相信它”“能否被消费者使用”“用了多少调用/Token/时间”。

## 六、正式知识文件格式建议

**设计判断。** YAML front matter 适合放稳定元数据和治理状态；正文应该让人无需解析 JSON 即能阅读、审阅和比较；机器完整结构和调用轨迹应作为独立文件或链接附属产物。

建议最小样例如下（字段名为 STKB 候选合同，不是外部标准）：

```markdown
---
knowledge_object_id: ko-sales-001
revision: 3
domain: sales
module: product_fact
object_type: PRODUCT_FACT
status: reviewing
source_document_ids:
  - doc-example-001
source_anchors:
  - doc-example-001#section-2
rule_version: sales-rules-vNEXT
content_schema_version: object-content-vNEXT
evidence_policy: field_level
generated_at: 2026-08-28T10:00:00+08:00
---

# 产品事实：服务范围与限制

## 定义

这是一项关于某产品服务范围的可复用事实，适用于指定版本和渠道。

## 适用范围

| 维度 | 内容 |
| --- | --- |
| 产品/版本 | 示例产品 / v2 |
| 客群或场景 | 首次咨询；未说明的客群不自动扩展 |
| 生效条件 | 以来源材料中的条件为准 |

## 核心事实

| 字段 | 内容 | 证据 |
| --- | --- | --- |
| 服务范围 | 原文支持的服务范围 | `doc-example-001#section-2` |
| 限制与排除 | 原文明确的限制；未说明处不补写 | `doc-example-001#section-2` |

## 销售消费提示

- 当客户询问服务范围时，可先确认产品版本和适用场景。
- 不得把未说明的条件扩展成承诺。

## 来源证据

- `doc-example-001#section-2`，全文字符 `18320-18408`：
  > 原文逐字摘录放在此处，不改写。
  - 支持字段：`服务范围`、`限制与排除`
  - evidence_kind：`direct_statement`

## 关联与归并

- 关联对象：`ko-sales-007`（同一产品版本的流程说明）
- 合并理由：共同产品版本和生效范围；字段可独立更新时仍保持对象分离。

## 审阅状态

- 当前状态：`reviewing`
- 未决项：渠道适用范围未在本资料中说明。
- 机器结构：见同名 `.json` sidecar；运行调用明细见运行记录链接。
```

正文可使用表格和列表，但不要把 `content` 整体序列化为 ` ```json `。若必须同时提供机器结构，优先使用：

```text
ko-sales-001.md       # 人读、可审阅、可引用
ko-sales-001.json     # 与同一 revision/fingerprint 对齐的机器结构
run-<id>/calls/*.json # 每次调用和阶段结果
```

Markdown、JSON sidecar 和数据库登记必须共享 `knowledge_object_id + revision + schema/rule fingerprint`，并能从正文的证据锚点回到原始 `DocumentPackage`。

## 七、对上一份调研的保留、降级与删除建议

| 处理 | 上一份调研结论/做法 | 本补充的处理理由 |
| --- | --- | --- |
| 保留并强化 | 证据需要精确摘录、字符区间、字段支持关系；`AtomicClaim → 对象化 → 硬校验 → 归一/归并 → 正式化` 分层；图投影不反向决定身份。 | LangExtract、GraphRAG、DocEE/DREEAM 和 W3C PROV 都提供了相应的一手依据；补充全局偏移、跨 chunk 证据链和派生关系。 |
| 保留 | Schema validity、证据定位/支持、对象 precision/recall、模块 macro-F1、字段覆盖、过拆/欠拆、冗余、稳定性、归并准确率等指标。 | 这些指标能把“结构合法”和“业务可用”分开；新增 cross-unit evidence、token、调用数和阶段耗时。 |
| 降级为实验假设 | “两阶段主张→对象”是默认推荐；固定 12—20 份样本、20% 双标、特定 F1/过拆阈值。 | 两阶段有 GraphRAG/长文档研究的工程依据，但 2—4 次和具体阈值不是外部定理；先冻结实验预算和样本后测量。 |
| 降级为可选对照 | LangExtract、GraphRAG、LlamaIndex 的具体框架或实现方式。 | 可借鉴中间账本、Schema 路径和证据可视化，不应引入完整框架或把其对象/图身份照搬到 STKB。 |
| 删除/禁止当作质量证明 | “每 3500 字一个对象”“每模块至少一个对象”“summary 长度/最小字符数足以证明完整”“Gold 10/10 即体系正确”。 | 外部文档级研究强调跨句论元、细粒度证据和对象边界；长度、模块覆盖和单一 Gold 不能替代业务 adjudication。 |
| 重新审查 | 现有 D1—D5/22 模块、保险术语、对象类型白名单以及任何把药品/投保/核保/理赔作为通用定义的规则。 | APQC/BPMN/DMN 和 CRM 厂商资料支持“通用骨架 + 作用域扩展”，不支持用保险资料固定通用销售本体。 |

## 八、最小验证实验与验收标准

### 8.1 实验矩阵

用同一批脱敏资料、同一模型版本、同一规则/Schema 版本、同一人工 Gold，至少比较三种编排：

| 方案 | 结构 | 适用假设 |
| --- | --- | --- |
| A：全文单次 | 1 次全文结构化发现/对象输出 + 程序校验；必要时 1 次定向修复 | 短文档、对象数量和输出长度可控 |
| B：两阶段 | 1 次全文主张/证据发现 + 1 次对象化/全局合并；必要时 1 次修复 | 当前常见文档能完整放入第一阶段上下文 |
| C：自适应（推荐验证） | 先判断 token/结构风险；短文档走 B，超长文档按 chunk discovery，再统一对象化；最多 2 次定向修复 | 低调用预算优先，同时保留长文档质量兜底 |

每个方案至少重复 3 次，抽取短/中/长文档各一组，并固定记录：

- 每个模型调用的 purpose、输入范围、Prompt/Schema 版本、输出、重试、错误、input/completion/total token 和 wall time；
- 阶段中间结果和最终对象的 claim 覆盖、证据定位、模块/对象类型、对象边界及冲突；
- 正式 Markdown 是否可读，机器 sidecar 是否与 `id/revision/fingerprint` 一致；
- 质量、成本、延迟和稳定性，不只比较最终对象数量或 Gold 分数。

### 8.2 首轮验收门槛（建议值）

以下是 STKB 的试验门槛，不是外部项目承诺；必须在实验前冻结：

1. 当前标准长度资料的模型调用 `p95 ≤ 4`，不出现“每个对象一次调用”。
2. 最终响应 Schema validity 为 100%；拒答、截断、解析失败均可定位到调用记录。
3. 正式对象的字段级证据定位率为 100%，高风险事实/限制字段无无证据写入。
4. 对象 precision/recall、模块 macro-F1 和过拆/欠拆相对冻结的当前基线不下降超过 5 个百分点；若基线本身未获业务认可，只用于回归对照，不作为验收结论。
5. 总 token、p50/p95 延迟与当前 32 次运行基线有明确对比；首轮目标为 token 不超过基线 60%，p95 wall time 不超过基线 1.25 倍，超出需给出调用/规则上下文原因。
6. Markdown 正文无需解析 JSON 即能回答“这是什么、适用什么、限制是什么、谁消费、证据在哪里、何时复核”；sidecar/运行记录可供机器和 Web 展开。

## 九、研究限制与未决问题

- 本次来源以 IE 论文、W3C/OMG 标准、官方项目/厂商文档为主；没有发现统一的跨行业销售知识本体，结论应表述为“当前检索未发现”，不能扩大为绝对不存在。
- 论文中的事件、关系或新闻文档任务与 STKB 销售资料不同，能支持的是抽取分层、跨句证据、评估方法和成本意识，不能直接提供模块名或字段名。
- 当前 32 次调用、Gold 10/10 和现有规则版本属于 STKB 项目运行/实现证据，不是外部研究结论，也不是重构后体系的验收证明。
- `strategy`、`playbook`、`script`、`case` 等销售语义是否是独立对象，仍需结合跨行业样例、消费者和生命周期审查；本笔记只提供 BPMN/DMN 和流程分类的边界参照。
- 2—4 次调用必须用同一资料实测 token、延迟、质量和稳定性后再冻结为实现合同；在此之前不得把它写成已完成能力。

## 十、来源清单

1. [UIE：Unified Structure Generation for Universal Information Extraction](https://aclanthology.org/2022.acl-long.395/)
2. [OneIE：A Joint Neural Model for Information Extraction with Global Features](https://aclanthology.org/2020.acl-main.713/)
3. [Document-level Event Extraction with Efficient End-to-end Learning of Cross-event Dependencies](https://aclanthology.org/2021.nuse-1.4/)
4. [DocEE：A Large-Scale and Fine-grained Benchmark for Document-level Event Extraction](https://aclanthology.org/2022.naacl-main.291/)
5. [DocEE-zh：A Fine-grained Benchmark for Chinese Document-level Event Extraction](https://aclanthology.org/2024.findings-emnlp.35/)
6. [DREEAM：Guiding Attention with Evidence for Improving Document-Level Relation Extraction](https://aclanthology.org/2023.eacl-main.145/)
7. [PICARD：Parsing Incrementally for Constrained Auto-Regressive Decoding](https://aclanthology.org/2021.emnlp-main.779/)
8. [OpenAI：Introducing Structured Outputs in the API](https://openai.com/index/introducing-structured-outputs-in-the-api/)
9. [JSON Schema Specification](https://json-schema.org/specification)
10. [Google LangExtract README](https://github.com/google/langextract/blob/main/README.md)
11. [Google LangExtract 长文档示例](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)
12. [Microsoft GraphRAG Methods](https://microsoft.github.io/graphrag/index/methods/)
13. [Microsoft GraphRAG Indexing Dataflow](https://microsoft.github.io/graphrag/index/default_dataflow/)
14. [Microsoft GraphRAG Outputs](https://microsoft.github.io/graphrag/index/outputs/)
15. [W3C PROV-DM](https://www.w3.org/TR/prov-dm/Overview.html)
16. [OMG BPMN 2.0.2](https://www.omg.org/spec/BPMN/2.0.2/PDF/)
17. [OMG DMN 1.4](https://www.omg.org/spec/DMN/1.4/PDF)
18. [APQC Cross-Industry Process Classification Framework](https://www.apqc.org/resource-library/resource-listing/apqc-process-classification-framework-pcf-cross-industry-pdf-7)
19. [Microsoft Dynamics 365 Sales Process Overview](https://learn.microsoft.com/en-us/dynamics365/project-operations/sales/sales-overview)
20. [Salesforce：Make Selling Simple with Sales Stages](https://help.salesforce.com/s/articleView?id=sf.essentials_sales_stages.htm&language=en_US)

## 十一、面向多 AI 应用的知识底座结构复审

> 本章针对“STKB 要为助手、对话分析、角色扮演、评估和场景包提供共享检索底座”的前提新增。它补充的是分类、对象合同、关系和消费方式的证据，不为 STKB 规定域、模块或对象合同的固定数量。外部资料中的模型、字段和组件名称也不应直接变成 STKB 合同。

### 11.1 先区分五件事：目录、内容、关系、投影和资产

**本地业务前提。** STKB 的价值不只是把一份资料切成若干可读文件，而是让不同 AI 应用从同一份可追溯知识中取不同的上下文。例如，助手回答产品问题时需要事实、适用范围和证据；对话分析从客户表达出发，需要找到需求、相关产品事实、建议策略和约束；角色扮演需要把客户画像、需求、产品、旅程、表达和合规条件装配成场景上下文；评估需要把行为证据与能力标准、量规及关键限制关联起来。每个应用的入口不同，但不应各自复制一套产品、客户或话术知识。

因此，STKB 至少要保留以下语义层，但它们不是五层都必须形成独立存储表：

| 层 | 解决的问题 | 建议语义 | 不应承担的职责 |
| --- | --- | --- | --- |
| 域（domain） | 读者和系统从哪个大的语义区域开始找 | 稳定的导航、归属、治理和检索 facet；例如供给、买方、销售指导、互动、约束评价只是候选命名 | 不做行业名、不做唯一的检索入口、不规定对象必须只能连接同域对象 |
| 知识内容模块（module） | 一组内容由谁维护、回答哪类业务问题、按什么方式消费 | 比域更细的语义责任单元；模块内可以有多个对象类型 | 不因字段列表不同就拆成新模块，不直接代表某个应用页面或场景包 |
| 对象合同（由 `objectType` 标识） | 一条 KnowledgeObject 应采用什么边界、身份和字段 Shape | `objectType` 是 KnowledgeObject 的属性，用于选择事实、规则、过程、画像、需求、策略、响应、量规等合同 | 不把对象合同画成模块下的新目录层，不用类型名称替代字段语义和证据 |
| 关系（relation） | 一个对象如何到达另一个对象 | 一等的、可约束、可追溯的跨域/跨模块边；例如 `addresses`、`supports`、`constrained_by`、`evaluated_by` | 不把关系隐含在摘要文字或模块名称中，不以向量相似度冒充业务关系 |
| 检索视图/应用装配（view/profile/bundle） | 某个应用在一个任务中需要哪些对象、字段和关系 | 对共享对象的筛选、扩展、排序、字段投影和提示上下文；场景包与评估集属于此层的候选资产 | 不复制并改写 canonical KnowledgeObject，不成为新的事实来源 |

这一区分是 STKB 的设计判断。它允许“从任何一个点到达其他基础知识”，同时把可复用真值与应用所需的上下文装配分开。一个抽象路径可以是：

在后续合同中，应用层建议固定区分两个名字：`RetrievalView` 是面向一次查询或一种任务的可重建检索投影，保存筛选条件、字段投影、排序/扩展策略和安全边界；`KnowledgePackage` 是面向一个场景、训练任务或评价任务的可版本化装配，保存对象 ID、对象修订、允许的 `TypedRelationship` 路径和应用参数。二者都引用 canonical `KnowledgeObject`，不复制事实正文。`TypedRelationship` 则必须有关系类型、方向、两端对象类型/模块约束、作用域和证据要求，不能只保存一个无语义的 `related_ids` 数组。

```text
客户表达/问题
  → 买方意图或需求
  → 相关供给事实与适用规则
  → 可选销售策略/方法
  → 可用响应或话术
  → 合规约束与评价量规
```

路径上的每个节点仍然属于自己的域和模块；跨节点连接依靠有类型的关系；应用决定从哪个节点开始、允许扩展几跳以及需要哪些字段。这样“模块合并”不会破坏应用之间的连接，反而可以减少同义对象和重复内容。

### 11.2 外部一手资料说明了什么

下表只记录来源直接支持的事实，以及对 STKB 可借鉴的边界。表中的“可用原则”是结合本项目目标的设计判断，不是外部标准对 STKB 的规定。

| 来源 | 外部事实 | 对 STKB 可用的原则 | 适用边界 |
| --- | --- | --- | --- |
| [W3C SKOS](https://www.w3.org/TR/skos-reference/) | SKOS 用 concept scheme、concept、标签、层级关系、关联关系、collection 和 mapping 组织知识组织系统；它明确把这类层级视为便于导航和找相关资源的组织结构，而不是关于世界的形式逻辑事实。 | 域和模块可以作为可维护的导航/分类体系；对象事实和跨对象关系应另行建模。域保留的价值在于可导航、可筛选、可治理，不在于让所有业务推理都依赖树结构。 | SKOS 不给销售领域规定域名、模块名或数量；也不能替代对象事实、规则推理或证据合同。 |
| [W3C OWL 2](https://www.w3.org/TR/owl-overview/) | OWL 提供具有形式语义的 classes、properties、individuals 和 data values；本体和实例可以用 RDF 交换。 | KnowledgeObject 采用的合同、字段和关系应与域/模块目录职责分开；不要把“目录中的分类项”直接等同于“业务对象类”或“事实”。 | OWL 的表达能力和推理成本不意味着 STKB 当前就要实现完整本体；对象身份、来源和版本仍需项目合同。 |
| [W3C SHACL](https://www.w3.org/TR/shacl/) | SHACL 将待检查的 data graph 与描述约束的 shapes graph 分开，验证后产生带 focus node、路径、严重级别和约束来源的 validation report。 | 由 `objectType` 选择的字段 Shape 和质量报告应与知识内容分离；这支持合并模块后仍保留合同差异，也支持把“结构合法”和“业务有证据”分开。 | SHACL 主要解决结构验证，不证明文本主张为真，也不决定对象应归哪个模块。 |
| [W3C PROV-DM](https://www.w3.org/TR/prov-dm/Overview.html) | PROV-DM 用 domain-agnostic 的 entity、activity、agent、derivation、revision 等表达数据如何产生、使用和派生，并允许应用扩展。 | 文档、原文片段、模型调用、候选对象、正式修订和投影文件属于来源/运行/派生层；它们不应被塞进销售域模块，也不应被误当作销售事实。 | 不需要照搬完整 PROV 图；需要保留的是来源、生成活动、修订和派生的可追踪语义。 |
| [Schema.org 词汇工作方式](https://schema.org/docs/howwework.html) | Schema.org 将术语分为 type、property 和 enumerated value；类型形成导航层，同时维护人读定义和“不以人阅读为目的”的机器文件；术语可以在 core 与 hosted extension 之间移动。 | STKB 应分开维护通用核心、行业扩展、对象合同和字段词典；机器结构与人读知识文件也应分开。保险术语可作为扩展，不应反向定义核心域。 | Schema.org 是通用 Web 词汇，不是销售知识本体；“可移入 core/extension”是演化经验，不是 STKB 的迁移方案。 |
| [Microsoft GraphRAG 数据流](https://microsoft.github.io/graphrag/index/default_dataflow/) 与 [输出模型](https://microsoft.github.io/graphrag/index/outputs/) | GraphRAG 把 Document、TextUnit、Entity、Relationship、Claim/Covariate、Community 和 Community Report 分成不同的中间或派生结构；TextUnit 与实体/关系相连，输出表可供后续检索和摘要。 | 共享底座应保留 canonical 对象、原文单元、关系和面向检索的派生视图；“社区报告/摘要/问答上下文”不应覆盖原始对象真值。 | GraphRAG 的实体、社区和摘要是其检索方案，不代表 STKB 必须采用同样的对象类型或图算法。 |
| [Neo4j GraphRAG](https://neo4j.com/labs/genai-ecosystem/graphrag/) | Neo4j 将向量检索与图结构检索结合，通过文档、块、实体、关系和聚类扩展上下文，并强调结构化查询能够提供更详细的检索轨迹和结果来源。 | 关系应是检索可用的结构，不是仅供展示的附属字段；回答、分析和评估都应能回到扩展过的对象及证据。 | 这是厂商的 GraphRAG 模式，不证明任何图关系都值得建立；关系必须经过类型、方向、适用范围和来源审查。 |
| [LlamaIndex PropertyGraphIndex](https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/) | Property Graph 由带标签和属性的节点及关系组成；其 `SchemaLLMPathExtractor` 可以限定实体、关系及允许的连接；查询时可以组合多个 sub-retriever，也可把图存储与外部向量存储组合。 | canonical graph 的对象类型/关系合同与应用检索器可以分开；同一底座可为不同应用组合不同检索器、字段和扩展路径。 | LlamaIndex 的 extractor、retriever 和 schema 是框架实现，不是 STKB 的领域分类；其默认行为仍需用本地证据和质量门禁校验。 |
| [Haystack Document Store](https://docs.haystack.deepset.ai/docs/document-store)、[Metadata Filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering) 与 [MultiRetriever](https://docs.haystack.deepset.ai/docs/multiretriever) | Haystack 将 Document Store 与 Retriever 分开；文档可以带 metadata 和 ID，查询时可按 metadata 过滤；MultiRetriever 可以并行使用多个检索器、合并去重，并在运行时选择 active retrievers。 | 物理存储/索引不等于业务模块；应用可以在共享对象库上按域、模块、行业、版本、场景和任务过滤或组合检索器，而不需要复制知识。 | MultiRetriever 当前文档标注为 experimental；只能借鉴“存储与检索装配分离”的原则，不把框架组件直接作为 STKB API。 |
| [OMG BPMN 2.0.2](https://www.omg.org/spec/BPMN/2.0.2/PDF/) 与 [OMG DMN 1.4](https://www.omg.org/spec/DMN/1.4/PDF) | BPMN 表达活动、事件、网关和顺序流；DMN 表达输入、决策、业务知识模型和知识来源，并可与流程协同。 | 流程、业务规则、销售策略和来源可以在同一域/模块目录下保留不同对象类型；模块是否合并应看消费者和生命周期，而不是只看字段差异。 | BPMN/DMN 是建模语言，不是 STKB 文件格式，也不自动解决销售策略的开放性和证据问题。 |

这些来源共同支持的是“职责分开但可组合”，不是增加目录层级：域/模块提供可发现性，KnowledgeObject 保存对象语义，类型合同约束字段，关系支持跨域导航，应用视图负责任务特化，来源记录负责审计。分类、对象、校验、检索投影和运行证据不能混成同一个字段集合。

### 11.3 三种可比较的组织模式

#### 模式 A：保留域和合并后的知识内容模块，再用关系图连接（推荐作为评审基线）

```text
Domain
  └─ KnowledgeModule
       └─ KnowledgeObject
KnowledgeObject ──uses──> ObjectType Contract / Shape
KnowledgeObject ──typed relation──> KnowledgeObject
ApplicationProfile / View / Bundle ──selects & expands──> Objects + Relations
Evidence / Provenance ──supports──> Object fields and relations
```

域用于导航、归属、作用域和治理；模块是比域更细的业务责任单元，但模块数量和边界通过跨行业样例审查后再定；KnowledgeObject 是实际知识主体，通过 `objectType` 选择合同以表达结构差异；关系允许产品、买方、销售指导、互动和评价互相到达；应用 profile 决定一次检索需要哪些对象和路径。

优点是共享知识只维护一份，应用可以从不同入口取相同对象，模块合并不会丢失语义导航。风险是域/模块可能重新变成僵硬的树，因此需要允许关系跨域、对象有辅助 facet，并禁止把域层级当作唯一推理依据。

#### 模式 B：扁平对象图，域和模块只作为标签或 facet

```text
KnowledgeObject ──uses──> ObjectType Contract / Shape
KnowledgeObject ──typed relation──> KnowledgeObject
Domain / Module = metadata tags
ApplicationProfile = query filter + graph expansion
```

这种模式最大限度减少目录约束，适合对象合同和关系尚未稳定的探索期。代价是治理者、检索器和 Web 页面都要自己解释标签，容易出现同义标签、归属漂移和跨应用的筛选逻辑不一致。它不是“没有层级”，而是把域/模块降为 metadata；如果没有统一词汇和治理规则，实际复杂度会转移到每个应用。

#### 模式 C：以应用视图/场景包为中心，底层引用共享对象

```text
Application / UseCase
  └─ RetrievalView / ScenarioPackage / EvaluationSet
       └─ references KnowledgeObject IDs and allowed relation paths
KnowledgeObject graph remains the canonical source
```

该模式最贴近应用交付：助手有问答视图，对话分析有对话证据视图，陪练有场景上下文包，评估有评价集。它可以明确每个应用要哪些字段、路径和安全边界，但如果把视图中的文字复制成新知识，就会形成多份真值和同步问题。因此视图、场景包和评估集只能保存引用、筛选、排序、应用特化说明和版本锁定；事实内容仍从 canonical 对象读取。

| 比较维度 | 模式 A：域/模块/对象/关系/视图 | 模式 B：扁平对象图 | 模式 C：应用视图中心 |
| --- | --- | --- | --- |
| 多应用复用 | 强，目录和关系都可复用 | 中，依赖各应用理解标签和关系 | 强，但依赖 canonical 引用不复制 |
| 跨域检索 | 关系显式，入口清晰 | 关系显式，但入口和范围较弱 | 可控，视图可限制路径；全局探索较弱 |
| 治理和归属 | 较强，域/模块可承载责任边界 | 较弱，需要额外的标签治理 | 视图治理强，底层语义治理仍需补足 |
| 合并 22 模块的影响 | 可合并内容模块而保留导航层 | 迁移简单，但语义责任容易退化为标签 | 应用适配快，但会掩盖 canonical 模块问题 |
| 主要风险 | 层级僵化或把模块当物理分库 | 标签漂移、每个应用重复解释 | 视图复制知识、应用孤岛化 |

**选择判断。** 对 STKB 当前目标，模式 A 应作为总体评审基线，模式 C 作为应用交付层，模式 B 作为底层存储的实现选项，而不是对外暴露的完整语义模型。这个选择并不意味着保留当前 D1—D5 或 22 个模块；它只意味着不要为了合并内容模块而丢掉域/模块作为共享导航和治理语义。

### 11.4 对“保留域 + 合并知识内容模块”与“取消层级”的证据化判断

| 判断问题 | 保留域并合并模块 | 取消域/模块层级 | 本项目判断 |
| --- | --- | --- | --- |
| 多应用是否需要共同入口 | 域/模块可以提供稳定 facet、默认检索范围和人读导航 | 每个应用自行定义标签和入口 | 保留更能避免助手、陪练和评估各自形成不兼容目录；域不应成为唯一入口 |
| 是否支持从任一点横向到达其他知识 | 关系作为一等语义跨域连接，模块只是起点和归属 | 同样可以有关系，但缺少共享的语义导航 | 关系图必须保留；取消层级不会自动解决关系问题 |
| 模块重复是否会继续存在 | 可以把重复内容合并为对象类型、公共字段或关系，保留模块责任边界 | 表面上消除了重复模块，实际可能把同义概念变成重复标签/视图逻辑 | 合并内容模块是必要动作，但不能把“去重”误解为“去掉所有目录” |
| 生命周期、证据和治理是否可区分 | 模块可以承载默认消费者、更新责任和来源要求，对象类型处理结构差异 | 需要在每个对象或每个应用 profile 中重复声明 | 保留模块，但模块合同必须写清责任、消费者、生命周期和证据；不靠名称存在 |
| 是否容易被保险样例固化 | 只要域/模块语义用通用词，保险概念挂在行业扩展和作用域下即可 | 扁平化并不会自动去掉保险字段，反而可能让样例标签直接扩散 | 核心问题是通用语义和扩展边界，不是层级数量 |

**结论。** 目前证据支持保留“域—知识内容模块”作为轻量的共享语义目录，同时合并那些业务问题、消费者、生命周期、证据和归属无法区分的现有模块；不支持取消这两层，也不支持把现有 22 个模块原样保留。更准确的目标是：

1. 域保留为可导航、可筛选、可治理的高层语义集合，具体域名和数量待样例审查；
2. 知识内容模块按独立业务责任合并重构，不按现有模块数倒推；
3. KnowledgeObject 通过 `objectType` 选择合同来解决结构差异，公共 `scope`、`evidence`、`relationships` 等字段避免重复；`objectType` 不构成目录层级；
4. 关系图承担跨域可达性，任何应用都可以从任意对象开始检索并按类型扩展；
5. `RetrievalProfile`、`ScenarioPackage`、`EvaluationSet` 等只装配和引用 canonical 对象，不成为第二套知识真值；
6. 物理上可以采用扁平表、property graph 或多个投影，但对外语义只保留域、模块、KnowledgeObject 和关系；`objectType` 作为 KnowledgeObject 的合同标识存在。

这是一项设计判断，不是外部标准规定的答案。外部资料只能说明这种分工与成熟知识组织、图检索和 RAG 编排方式相容；模块名称、合并范围和关系类型必须由 STKB 自己的样例及消费者验收决定。

### 11.5 用应用消费反推模块和关系，而不是从模块反推应用

| 应用场景 | 典型检索起点 | 需要横向到达的知识 | 视图/装配层应控制什么 | 不能把什么写回 canonical 对象 |
| --- | --- | --- | --- | --- |
| 助手：知识问答 | 用户问题、术语或产品/服务实体 | 事实、适用范围、过程/规则、相关响应和来源证据 | 结果数量、版本/行业/渠道过滤、证据展开和人读摘要 | 不因一次问答生成新的产品承诺或把回答当事实 |
| 对话分析 | 一句话、会话轮次、客户表达 | 意图、需求、信号、相关供给事实、建议动作、合规限制 | 证据窗口、对话阶段、允许扩展的关系和输出标签 | 不把模型对客户的推断直接写成稳定画像；不把建议动作写成事实 |
| 角色扮演/陪练 | 场景、角色、目标或客户画像 | 买方画像/需求、供给事实、销售旅程、策略/方法、表达模板、约束 | 场景范围、角色可见信息、难度、允许/禁止知识、引用的对象修订 | 不复制一份产品/话术正文；不把运行时对话状态写入知识对象 |
| 评估 | 会话、行为证据或评价任务 | 能力标准、量规、关键规则、期望/禁止表现、相关事实 | 评分维度、证据窗口、阈值和报告格式 | 不把一次评分结果改写成能力标准或销售规则 |
| 场景包 | 场景定义或训练任务 | 可见的对象集合及其关系子图，必要时连接产品、画像、话术和评价 | 引用的对象 ID/修订、关系路径、覆盖范围和应用参数 | 不把场景包当作通用事实模块；不与对象正文分叉维护 |

这张表给模块审查提供了反向检验：一个候选模块必须能说明哪些应用会直接消费它、哪些应用只通过关系间接消费它、它的更新会影响哪些视图，以及它与相邻模块是否确实存在不同的身份和生命周期。如果两个模块总是被同一应用同时读取、由同一责任方维护、用同一来源更新，且拆开后没有独立的检索或审阅价值，就应进入合并候选；如果只是应用显示方式不同，应建立视图而不是新模块。

### 11.6 本地必须验证的事项

外部资料不足以确认 STKB 的实际结构。下一轮应在不连接数据库、不运行真实模型的前提下先完成以下静态和人工验证：

1. 用 SaaS、制造服务、金融服务和保险各一组资料，重新标注 `domain/module/object_type/relation`，记录每一次合并、拆分和暂不确定的理由；保险词汇只放在行业扩展或样例作用域中。
2. 为助手、对话分析、陪练、评估和场景包各建立最小检索任务集，写出起点对象、期望到达的对象、允许的关系路径、不得跨越的边界和所需证据；至少覆盖“从产品到话术/画像/维度”和“从客户表达回到产品/规则”的双向路径。
3. 对当前 22 个模块计算候选合并证据：业务问题是否相同、默认消费者是否相同、身份/生命周期是否相同、来源和审阅责任是否相同、是否可由同一 KnowledgeObject 合同表达。不能只凭名称相似或字段重叠合并。
4. 对模式 A、B、C 使用同一批人工标注对象做离线检索对比，至少测对象召回、跨域路径准确率、错误扩展率、孤立对象率、重复内容率和视图重建一致性；不要用“模块命中数”代替这些指标。
5. 验证一个对象修改后的影响面：是否能通过 ID/修订找出受影响的助手视图、对话分析规则、陪练场景包和评价集；若必须手工复制正文，说明分类或关系合同仍不清楚。
6. 在结构稳定前，不修改当前 Gold 使其迎合新模块；先保存旧 Gold 的来源和结果，再建立包含应用消费路径、对象边界和证据锚点的新 Gold 版本。

本章的结论停留在“保留域/模块分类、合并重复内容、以对象合同约束结构、以关系和应用视图实现复用”的评审基线。具体域数、模块数、对象合同、关系白名单和数据库投影仍未通过本地样例验收，不能表述为已完成的重构。

### 11.7 “5 域 + 显著合并模块”的评审标准

如果下一轮仍以 5 个域作为候选导航骨架，5 只能是待验证的候选结构，不是实现约束。评审时应逐域给出证据，至少满足以下条件：

| 评审项 | 通过标准 | 不通过时的处理 |
| --- | --- | --- |
| 跨行业稳定性 | 在至少两类非保险行业中，域名表达的业务语义仍成立；行业术语只能成为扩展或作用域 | 改名、降级为标签，或移入行业扩展 |
| 多应用入口价值 | 至少两个 AI 应用会以该域作为起点、过滤条件或治理归属，并能解释起点不同但对象可互联 | 不建立域级概念，只保留模块/对象标签 |
| 语义同质性 | 域内模块共享一组可解释的上位问题，但不要求共享所有字段 | 拆出语义明显不同的模块，避免用域名掩盖混杂 |
| 跨域可达性 | 域允许通过 `TypedRelationship` 连接其他域，不把树路径当作唯一事实路径 | 补关系合同；禁止把跨域关系写成摘要中的隐含语句 |
| 生命周期与治理 | 能说明域内内容的归属、版本和审阅入口；治理边界不依赖保险流程词汇 | 重新定义归属或把治理放到对象/模块层 |
| 可迁移性 | 旧 22 模块能逐项映射到候选域/模块，无法映射的项有明确的视图、资产或淘汰理由 | 暂不改合同，先建立迁移清单和待决项 |

“显著合并模块”也不能用“数量从 22 变成多少”证明。两个或多个现有模块只有在以下证据同时成立时才进入合并候选：

1. 它们回答的是同一个主要业务问题，且主要消费者高度重合；
2. 对象身份的主体、作用域、版本和更新触发基本一致；
3. 来源权威、审阅责任和生效/失效生命周期基本一致；
4. 合并后仍能用 `ObjectType` 和类型级字段合同表达结构差异，不需要在内容里写一段无法校验的自由文本；
5. 合并不会抹掉高风险约束、证据要求或关系语义；
6. 至少有一组跨行业样例证明合并后的上位语义不是保险专用词。

下列任一情况都足以阻止直接合并：不同权威来源或不同责任主体；不同生效/失效节奏；不同安全或合规等级；不同的对象身份和归并粒度；一个对象需要被多个消费者独立更新；或合并后只能靠 `kind`、`summary` 和最小字符数恢复原边界。此时应先检查是否应拆成同一域下的不同对象类型，或者把其中一项降为 `RetrievalView`/`KnowledgePackage`，而不是继续增加字段。

这组标准的依据是：SKOS 将层级与集合用于导航，OWL 将类/属性/个体用于形式内容语义，SHACL 将结构约束与数据图分开，GraphRAG/LlamaIndex/Neo4j/Haystack 则分别展示了中间对象、关系图、检索器和应用装配可以分开演化。它们支持“保留域/模块分类、合并重复责任、对象采用类型合同、关系连接、应用投影”的方向，但没有任何一项规定 STKB 必须有 5 个域或某个模块数量。
