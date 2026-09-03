<script setup lang="ts">
import {
  IconAlertCircle,
  IconBook2,
  IconCheck,
  IconCode,
  IconFileDescription,
  IconPlayerPlay,
  IconRefresh,
  IconStack2,
} from '@tabler/icons-vue'
import { computed, onMounted, ref, type Component } from 'vue'

import {
  formKnowledgeObjects,
  getKnowledgeFormation,
  getDocumentPackage,
  getIdentificationCatalog,
  IdentificationApiError,
  listSourceMaterials,
  listIdentificationRuns,
  runIdentification,
} from '../api'
import {
  formatDuration,
  prettyJson,
  type DocumentPackage,
  type FormalKnowledgeObject,
  type IdentificationCatalog,
  type IdentificationResult,
  type KnowledgeFormationResult,
  type KnowledgeModule,
  type ModelCallTrace,
  type ProcessingStage,
  type SourceMaterial,
} from '../types'

type MainView = 'build' | 'knowledge' | 'rules' | 'evidence'
type BuildPhase = 'idle' | 'ready' | 'recognizing' | 'forming' | 'review' | 'completed' | 'failed'

const viewItems: { key: MainView; label: string; icon: Component }[] = [
  { key: 'build', label: '构建流程', icon: IconPlayerPlay },
  { key: 'knowledge', label: '知识对象', icon: IconStack2 },
  { key: 'rules', label: '规则库', icon: IconBook2 },
  { key: 'evidence', label: '运行证据', icon: IconCode },
]

const activeView = ref<MainView>('build')
const buildPhase = ref<BuildPhase>('idle')
const sourceMaterials = ref<SourceMaterial[]>([])
const selectedMaterialId = ref('')
const documentPackage = ref<DocumentPackage | null>(null)
const catalog = ref<IdentificationCatalog | null>(null)
const identification = ref<IdentificationResult | null>(null)
const formation = ref<KnowledgeFormationResult | null>(null)
const selectedObjectId = ref('')
const selectedDomainCode = ref('D1')
const selectedModuleCode = ref('D1.1')
const selectedCallIndex = ref(0)
const error = ref('')
const isLoadingPrevious = ref(false)
const isLoadingMaterial = ref(false)

const selectedMaterial = computed(() =>
  sourceMaterials.value.find((item) => item.documentPackageId === selectedMaterialId.value) ?? null,
)

const selectedKnowledgeObject = computed<FormalKnowledgeObject | null>(() =>
  formation.value?.knowledgeObjects.find(
    (item) => item.knowledgeObjectId === selectedObjectId.value,
  ) ?? formation.value?.knowledgeObjects[0] ?? null,
)

const storageEvidence = computed(() => formation.value?.storageEvidence ?? null)

const selectedRelationships = computed(() => {
  const objectId = selectedKnowledgeObject.value?.knowledgeObjectId
  if (!objectId) return []
  return formation.value?.relationships?.filter(
    (item) => item.sourceRef === objectId || item.targetRef === objectId,
  ) ?? []
})

const selectedDomain = computed(() =>
  catalog.value?.domains.find((item) => item.code === selectedDomainCode.value) ?? null,
)

const domainModules = computed(() =>
  catalog.value?.modules.filter((item) => item.domain === selectedDomainCode.value) ?? [],
)

const selectedModule = computed<KnowledgeModule | null>(() =>
  domainModules.value.find((item) => item.code === selectedModuleCode.value)
    ?? domainModules.value[0]
    ?? null,
)

const selectedCall = computed<ModelCallTrace | null>(() =>
  identification.value?.modelCalls[selectedCallIndex.value] ?? null,
)

const repairCalls = computed(() =>
  identification.value?.modelCalls
    .map((call, index) => ({ call, index }))
    .filter(({ call }) => ['repair', 'output_limit_retry'].includes(call.purpose)) ?? [],
)

const buildSteps = computed(() => {
  const hasSource = Boolean(documentPackage.value)
  const hasRules = Boolean(catalog.value)
  const hasIdentification = Boolean(identification.value)
  const hasFormation = Boolean(formation.value)
  return [
    { key: 'source', label: '准备资料', detail: '读取全文与来源定位', done: hasSource },
    { key: 'rules', label: '装载规则', detail: `5个销售域 / ${catalog.value?.modules.length ?? 0}个模块`, done: hasRules },
    { key: 'recognize', label: '识别知识', detail: '发现并组织知识内容', done: hasIdentification },
    { key: 'validate', label: '校验结果', detail: '对象边界、分类与证据', done: hasIdentification },
    { key: 'merge', label: '归一与归并', detail: '实体身份与知识身份', done: hasFormation },
    { key: 'write', label: '形成知识对象', detail: '评审后登记正式版本', done: formation.value?.status === 'completed' },
    { key: 'project', label: '同步三种存储', detail: '正式文件、pgvector与Neo4j', done: Boolean(storageEvidence.value && !storageEvidence.value.errors.length && storageEvidence.value.pgvectorRecords) },
  ]
})

const currentStepKey = computed(() => {
  if (buildPhase.value === 'recognizing') return 'recognize'
  if (buildPhase.value === 'forming') return 'merge'
  if (!documentPackage.value) return 'source'
  if (!catalog.value) return 'rules'
  if (!identification.value) return 'recognize'
  if (!formation.value) return 'merge'
  if (formation.value.status === 'review_required') return 'write'
  if (formation.value.status === 'failed') return 'project'
  return ''
})

const phaseLabel = computed(() => {
  const labels: Record<BuildPhase, string> = {
    idle: '选择一份资料开始',
    ready: '资料和规则已就绪',
    recognizing: '正在识别并校验销售知识',
    forming: '正在形成知识对象并同步三种存储',
    review: '发现版本差异，等待质量评审',
    completed: '本次知识构建已完成',
    failed: '本次构建未完成',
  }
  return labels[buildPhase.value]
})

const canBuild = computed(() => Boolean(documentPackage.value && catalog.value))
const isBuilding = computed(() => ['recognizing', 'forming'].includes(buildPhase.value))

function errorMessage(reason: unknown): string {
  if (reason instanceof IdentificationApiError) return reason.message
  if (reason instanceof Error) return reason.message
  return '请求未完成，请检查服务日志。'
}

async function initialize(): Promise<void> {
  try {
    catalog.value = await getIdentificationCatalog()
  } catch (reason) {
    error.value = errorMessage(reason)
    buildPhase.value = 'failed'
    return
  }
  try {
    sourceMaterials.value = await listSourceMaterials()
  } catch (reason) {
    error.value = `资料运行记录暂不可用，仍可查看规则：${errorMessage(reason)}`
  }
}

async function selectMaterial(): Promise<void> {
  documentPackage.value = null
  identification.value = null
  formation.value = null
  selectedObjectId.value = ''
  error.value = ''
  if (!selectedMaterialId.value) {
    buildPhase.value = 'idle'
    return
  }
  isLoadingMaterial.value = true
  try {
    documentPackage.value = await getDocumentPackage(selectedMaterialId.value)
    buildPhase.value = 'ready'
    activeView.value = 'build'
  } catch (reason) {
    error.value = errorMessage(reason)
    buildPhase.value = 'failed'
  } finally {
    isLoadingMaterial.value = false
  }
}

function resetWorkbench(): void {
  selectedMaterialId.value = ''
  documentPackage.value = null
  identification.value = null
  formation.value = null
  selectedObjectId.value = ''
  selectedCallIndex.value = 0
  selectedDomainCode.value = 'D1'
  selectedModuleCode.value = 'D1.1'
  error.value = ''
  buildPhase.value = 'idle'
  activeView.value = 'build'
}

async function buildKnowledge(): Promise<void> {
  if (!documentPackage.value) return
  error.value = ''
  identification.value = null
  formation.value = null
  try {
    buildPhase.value = 'recognizing'
    const nextIdentification = await runIdentification(
      documentPackage.value.documentPackageId,
    )
    identification.value = nextIdentification
    if (nextIdentification.status !== 'completed') {
      throw new Error('销售知识识别失败，请在运行证据中查看错误。')
    }
    buildPhase.value = 'forming'
    const nextFormation = await formKnowledgeObjects(nextIdentification.runId)
    formation.value = nextFormation
    selectedObjectId.value = nextFormation.knowledgeObjects[0]?.knowledgeObjectId ?? ''
    selectedCallIndex.value = 0
    if (nextFormation.status === 'failed') {
      throw new Error('知识对象已形成，但存储投影失败，请查看存储证据。')
    }
    buildPhase.value = nextFormation.status === 'review_required' ? 'review' : 'completed'
    activeView.value = 'knowledge'
  } catch (reason) {
    error.value = errorMessage(reason)
    buildPhase.value = 'failed'
  }
}

async function loadLatestResult(): Promise<void> {
  if (!documentPackage.value) return
  error.value = ''
  isLoadingPrevious.value = true
  try {
    const runs = await listIdentificationRuns(documentPackage.value.documentPackageId, 1)
    const latest = runs.at(-1)
    if (!latest) throw new Error('这份资料还没有可载入的历史运行。')
    identification.value = latest
    formation.value = await getKnowledgeFormation(latest.runId)
    selectedObjectId.value = formation.value.knowledgeObjects[0]?.knowledgeObjectId ?? ''
    buildPhase.value = formation.value.status === 'review_required' ? 'review' : 'completed'
    activeView.value = 'knowledge'
  } catch (reason) {
    error.value = errorMessage(reason)
  } finally {
    isLoadingPrevious.value = false
  }
}

function selectDomain(domainCode: string): void {
  selectedDomainCode.value = domainCode
  selectedModuleCode.value = catalog.value?.modules.find(
    (item) => item.domain === domainCode,
  )?.code ?? ''
}

function objectActionLabel(action: FormalKnowledgeObject['action']): string {
  return { created: '待新增', updated: '更新', reused: '复用', review_required: '待评审' }[action]
}

function callPurposeLabel(purpose: ModelCallTrace['purpose']): string {
  return {
    identification: '历史单阶段识别',
    claim_discovery: '发现原子主张',
    object_planning: '规划对象边界',
    content_realization: '编制完整内容',
    object_formation: '历史对象形成',
    output_limit_retry: '输出续接',
    repair: '结构修复',
  }[purpose]
}

function callsForStage(stage: ProcessingStage): Array<{ call: ModelCallTrace; index: number }> {
  const calls = identification.value?.modelCalls ?? []
  if (stage.modelCallIds?.length) {
    const ids = new Set(stage.modelCallIds)
    return calls
      .map((call, index) => ({ call, index }))
      .filter(({ call }) => ids.has(call.callId))
  }
  return calls
    .map((call, index) => ({ call, index }))
    .filter(({ call }) => call.purpose === stage.key)
}

function callResultFlow(purpose: ModelCallTrace['purpose']): string {
  return {
    identification: '进入历史候选结果',
    claim_discovery: '输出进入主张证据校验',
    object_planning: '输出进入对象合同与覆盖校验',
    content_realization: '输出进入候选对象质量门禁',
    object_formation: '输出进入正式知识形成',
    output_limit_retry: '续接输出回到原调用结果',
    repair: '修复结果回到原阶段解析',
  }[purpose]
}

onMounted(() => {
  void initialize()
})
</script>

<template>
  <main class="knowledge-build-app">
    <header class="build-topbar">
      <div class="build-brand"><span>ST</span><div><strong>STKB</strong><small>销售知识构建验证</small></div></div>
      <nav aria-label="主视图">
        <button v-for="item in viewItems" :key="item.key" type="button" :class="{ active: activeView === item.key }" @click="activeView = item.key">
          <component :is="item.icon" size="16" stroke="1.8" />{{ item.label }}
          <b v-if="item.key === 'knowledge' && formation">{{ formation.knowledgeObjects.length }}</b>
        </button>
      </nav>
      <div class="build-topbar-actions"><button class="build-reset" type="button" :disabled="isBuilding || isLoadingPrevious || isLoadingMaterial" title="只清空当前页面状态，不删除已落盘的知识对象" @click="resetWorkbench"><IconRefresh size="15" />重置页面</button><div class="build-service"><i></i>服务可用</div></div>
    </header>

    <section class="build-shell">
      <div class="build-title-row">
        <div><p>能力验证 / 销售知识构建</p><h1>从业务资料形成可追溯的知识对象</h1></div>
        <div v-if="catalog" class="rule-brief"><span>当前规则</span><strong>{{ catalog.version }}</strong><small>5个销售域 · {{ catalog.modules.length }}个知识模块</small><button type="button" @click="activeView = 'rules'">查看规则</button></div>
      </div>

      <div v-if="error" class="build-error"><IconAlertCircle size="18" /><span>{{ error }}</span><button type="button" @click="initialize"><IconRefresh size="14" />重新检查</button></div>

      <section v-if="activeView === 'build'" class="build-view">
        <div class="build-command">
          <label><span>选择测试资料</span><select v-model="selectedMaterialId" :disabled="isBuilding || isLoadingMaterial" @change="selectMaterial"><option value="">请选择资料</option><option v-for="item in sourceMaterials" :key="item.documentPackageId" :value="item.documentPackageId" :disabled="item.status !== 'available'">{{ item.sourceFileName }}</option></select></label>
          <div v-if="documentPackage" class="source-inline"><IconFileDescription size="18" /><div><strong>{{ documentPackage.sourceFileName }}</strong><span>{{ documentPackage.anchors.length }}个来源位置 · {{ documentPackage.processingMethod === 'agent_assisted' ? '已完成代理解析' : '已完成解析' }}</span></div></div>
          <div class="build-actions"><button class="build-secondary" type="button" :disabled="!canBuild || isBuilding || isLoadingPrevious || isLoadingMaterial" @click="loadLatestResult"><IconRefresh size="15" />{{ isLoadingPrevious ? '载入中' : '载入最近结果' }}</button><button class="build-primary" type="button" :disabled="!canBuild || isBuilding || isLoadingMaterial" @click="buildKnowledge"><IconPlayerPlay v-if="!isBuilding" size="17" fill="currentColor" /><span v-else class="build-pulse"></span>{{ isBuilding ? phaseLabel : formation ? '重新构建' : '开始构建知识' }}</button></div>
        </div>

        <div class="flow-board">
          <div class="flow-status"><span :class="`phase-${buildPhase}`"></span><div><strong>{{ phaseLabel }}</strong><p v-if="selectedMaterial">{{ selectedMaterial.sourceFileName }}</p></div></div>
          <ol class="flow-steps">
            <li v-for="(step, index) in buildSteps" :key="step.key" :class="{ done: step.done, current: currentStepKey === step.key }"><span><IconCheck v-if="step.done" size="14" stroke="2.4" /><template v-else>{{ index + 1 }}</template></span><div><strong>{{ step.label }}</strong><small>{{ step.detail }}</small></div></li>
          </ol>
        </div>

        <div v-if="formation" class="build-outcome" :class="{ review: formation.status === 'review_required' }">
          <div><span>{{ formation.status === 'review_required' ? '本次待审' : '本次形成' }}</span><strong>{{ formation.knowledgeObjects.length }}</strong><em>个知识对象</em></div>
          <dl><div><dt>待新增</dt><dd>{{ formation.createdCount }}</dd></div><div><dt>待评审</dt><dd>{{ formation.reviewRequiredCount }}</dd></div><div><dt>质量阻断</dt><dd>{{ formation.qualityBlockedCount }}</dd></div><div><dt>直接复用</dt><dd>{{ formation.reusedCount }}</dd></div><div><dt>正式文件</dt><dd>{{ formation.formalKnowledgeFiles }}</dd></div></dl>
          <button type="button" @click="activeView = 'knowledge'">{{ formation.status === 'review_required' ? '审阅版本差异' : '查看知识结果' }}</button>
        </div>

        <div v-else class="build-guidance"><div><span>验证目标</span><h2>从DocumentPackage形成知识对象并同步三种存储</h2><p>一次执行会完成真实模型识别、证据校验、实体与知识身份归并、正式Markdown写入、pgvector检索投影和Neo4j图投影。</p></div><div class="build-boundary"><strong>同一修订点</strong><span>正式知识文件</span><span>pgvector检索投影</span><span>Neo4j图投影</span></div></div>
      </section>

      <section v-else-if="activeView === 'knowledge'" class="knowledge-result-view">
        <div v-if="formation && storageEvidence" class="storage-proof" :class="{ failed: storageEvidence.errors.length }"><article><span>正式知识</span><strong>{{ storageEvidence.postgresObjects }} 对象 / {{ storageEvidence.formalFiles }} 文件</strong><small>PostgreSQL登记与Markdown</small></article><article><span>向量检索</span><strong>{{ storageEvidence.pgvectorRecords }} 条</strong><small>{{ storageEvidence.embeddingModel || '未执行' }} · {{ storageEvidence.embeddingTokens }} tokens · {{ storageEvidence.vectorDurationMs }} ms</small></article><article><span>知识图谱</span><strong>{{ storageEvidence.neo4jKnowledgeObjects }} 对象 / {{ storageEvidence.neo4jRelationships }} 关系</strong><small>承载 {{ storageEvidence.neo4jDocumentLinks }} · 实体引用 {{ storageEvidence.neo4jEntityReferences }} · 知识关系 {{ storageEvidence.neo4jKnowledgeRelationships }} · {{ storageEvidence.graphDurationMs }} ms</small></article><p v-if="storageEvidence.errors.length">{{ storageEvidence.errors.join('；') }}</p><details><summary>查看实际存储记录</summary><div><section><h3>正式文件</h3><pre>{{ prettyJson(storageEvidence.formalRecords) }}</pre></section><section><h3>pgvector记录</h3><pre>{{ prettyJson(storageEvidence.vectorRecords) }}</pre></section><section><h3>Neo4j节点</h3><pre>{{ prettyJson(storageEvidence.graphNodes) }}</pre></section><section><h3>Neo4j关系</h3><pre>{{ prettyJson(storageEvidence.graphRelationships) }}</pre></section></div></details></div>
        <div v-if="formation" class="knowledge-result-grid">
          <aside class="knowledge-object-list"><div class="result-pane-title"><div><p>KNOWLEDGE OBJECTS</p><h2>{{ formation.status === 'review_required' ? '知识对象审阅' : '正式知识对象' }}</h2></div><span>{{ formation.knowledgeObjects.length }}</span></div><div class="object-list-scroll"><button v-for="item in formation.knowledgeObjects" :key="item.knowledgeObjectId" type="button" :class="{ active: selectedKnowledgeObject?.knowledgeObjectId === item.knowledgeObjectId }" @click="selectedObjectId = item.knowledgeObjectId"><span>{{ objectActionLabel(item.action) }}</span><div><strong>{{ item.title }}</strong><small>{{ item.domain }} / {{ item.module }} · {{ item.objectType }}</small></div></button></div></aside>
          <article v-if="selectedKnowledgeObject" class="formal-object-detail"><header><div><p>{{ selectedKnowledgeObject.knowledgeObjectId }} · revision {{ selectedKnowledgeObject.revision }}</p><h2>{{ selectedKnowledgeObject.title }}</h2><span>{{ selectedKnowledgeObject.domain }} / {{ selectedKnowledgeObject.module }} · {{ selectedKnowledgeObject.objectType }}</span></div><em>{{ objectActionLabel(selectedKnowledgeObject.action) }}</em></header><div class="formal-object-scroll">
            <section v-if="selectedKnowledgeObject.revisionProposal" class="revision-review">
              <div class="revision-review-intro"><div><span>版本差异</span><strong>正式版本尚未改写</strong><p>本次模型结果命中了同一来源维护单元，但正文有 {{ selectedKnowledgeObject.revisionProposal.changedPaths.length }} 处差异，需要判断是有效补充、事实冲突，还是单纯措辞漂移。</p></div><b>{{ selectedKnowledgeObject.revisionProposal.changedPaths.length }}</b></div>
              <div class="revision-columns"><article><header><span>当前正式版</span><strong>revision {{ selectedKnowledgeObject.revision }}</strong></header><pre>{{ prettyJson(selectedKnowledgeObject.content) }}</pre></article><article><header><span>本次建议</span><strong>尚未生效</strong></header><pre>{{ prettyJson(selectedKnowledgeObject.revisionProposal.content) }}</pre></article></div>
              <div class="changed-paths"><h3>发生变化的正文路径</h3><code v-for="path in selectedKnowledgeObject.revisionProposal.changedPaths" :key="path">{{ path }}</code></div>
            </section>
            <section v-else><div v-if="selectedKnowledgeObject.equivalenceReason" class="equivalence-note"><strong>沿用正式版本</strong><span>{{ selectedKnowledgeObject.equivalenceReason }}</span></div><h3>规范内容</h3><pre>{{ prettyJson(selectedKnowledgeObject.content) }}</pre></section>
            <section><h3>本次正文主张追溯</h3><div class="source-trace-list"><article v-for="trace in selectedKnowledgeObject.sourceTraces" :key="trace.candidateId"><header><strong>{{ trace.candidateId }}</strong><span>{{ trace.attributedContentLeafCount }}/{{ trace.contentLeafCount }} 个正文叶子已归因</span></header><div><code v-for="usage in trace.claimUsage" :key="`${trace.candidateId}-${usage.claimId}`">{{ usage.claimId }} → {{ usage.contentPaths.join('、') }}</code></div></article></div></section>
            <section><h3>正式知识关系</h3><div v-if="selectedRelationships.length" class="formal-refs"><div v-for="item in selectedRelationships" :key="item.relationshipId"><code>{{ item.sourceRef }}</code><strong>{{ item.relationType }}</strong><code>{{ item.targetRef }}</code></div></div><p v-else>该对象暂未形成已确认关系。</p></section><section><h3>业务实体引用</h3><div v-if="selectedKnowledgeObject.entityReferences.length" class="formal-refs"><div v-for="item in selectedKnowledgeObject.entityReferences" :key="`${item.entityId}-${item.referenceRole}`"><code>{{ item.entityId }}</code><strong>{{ item.referenceRole }}</strong><span>{{ item.evidence.join('、') }}</span></div></div><p v-else>该对象暂未形成业务实体引用。</p></section><section><h3>当前正式版来源证据</h3><div class="formal-evidence"><code v-for="item in selectedKnowledgeObject.evidence" :key="item">{{ item }}</code></div></section><section class="formal-file"><h3>正式知识文件</h3><code>{{ selectedKnowledgeObject.filePath }}</code><span>SHA-256 {{ selectedKnowledgeObject.fileSha256 }}</span></section></div></article>
        </div>
        <div v-else class="build-empty"><IconStack2 size="30" /><h2>还没有形成正式知识对象</h2><p>返回构建流程，选择资料并完成一次知识构建。</p><button type="button" @click="activeView = 'build'">返回构建流程</button></div>
      </section>

      <section v-else-if="activeView === 'rules'" class="rules-library-view">
        <div v-if="catalog" class="rules-library-grid">
          <nav class="rules-domain-list"><div><p>SALES DOMAINS</p><h2>销售域</h2></div><button v-for="domain in catalog.domains" :key="domain.code" type="button" :class="{ active: selectedDomainCode === domain.code }" @click="selectDomain(domain.code)"><span>{{ domain.code }}</span><div><strong>{{ domain.name }}</strong><small>{{ domain.question }}</small></div></button></nav>
          <div class="rules-module-list"><div><p>{{ selectedDomain?.code }}</p><h2>{{ selectedDomain?.name }}</h2><span>{{ domainModules.length }}个模块</span></div><button v-for="module in domainModules" :key="module.code" type="button" :class="{ active: selectedModule?.code === module.code }" @click="selectedModuleCode = module.code"><code>{{ module.code }}</code><div><strong>{{ module.name }}</strong><small>{{ module.scope === 'core' ? '核心范围' : '可选范围' }}</small></div></button></div>
          <article v-if="selectedDomain && selectedModule" class="rules-detail"><header><div><p>{{ selectedModule.code }}</p><h2>{{ selectedModule.name }}</h2></div><span>{{ catalog.version }} / {{ catalog.contentContractVersion }} / {{ catalog.identityContractVersion }}</span></header><div class="rules-detail-scroll"><section class="domain-definition"><b>{{ selectedDomain.question }}</b><strong>{{ selectedDomain.meaning }}</strong><p>{{ selectedDomain.boundary }}</p></section><section><h3>模块定义</h3><p>{{ selectedModule.meaning }}</p></section><section class="rule-highlight"><h3>KnowledgeObject 边界</h3><p>{{ selectedModule.contentContract.granularity }}</p></section><section class="contract-decision-grid"><div><h3>纳入条件</h3><p>{{ selectedModule.contentContract.inclusion }}</p></div><div><h3>排除条件</h3><p>{{ selectedModule.contentContract.exclusion }}</p></div></section><section><h3>对象身份字段</h3><div class="rule-type-list"><code v-for="item in selectedModule.identityContract.identityFields" :key="item">{{ item }}</code></div><p class="contract-minimum">只有这些稳定业务字段参与对象身份；摘要、模块码、来源锚点不参与。</p></section><section class="contract-decision-grid"><div><h3>何时归为同一对象</h3><p>{{ selectedModule.identityContract.sameObjectWhen }}</p></div><div><h3>何时必须拆分</h3><p>{{ selectedModule.identityContract.differentObjectWhen }}</p></div></section><section class="contract-decision-grid"><div><h3>归并方式</h3><p>{{ selectedModule.identityContract.mergeStrategy }}</p></div><div><h3>冲突裁决</h3><p>{{ selectedModule.identityContract.conflictRule }}</p></div></section><section><h3>对象合同字段</h3><p class="contract-minimum">objectType 只选择 KnowledgeObject 的字段与校验合同，不形成新的知识层级。</p><div v-if="Object.keys(selectedModule.contentContract.requiredFieldsByType).length" class="contract-decision-grid"><div v-for="(fields, objectType) in selectedModule.contentContract.requiredFieldsByType" :key="objectType"><h3>{{ objectType }}</h3><div class="rule-type-list"><code v-for="field in fields" :key="field">{{ field }}</code></div></div></div><div v-else class="rule-type-list"><code v-for="item in selectedModule.contentContract.requiredFields" :key="item">{{ item }}</code></div><div v-if="Object.keys(selectedModule.contentContract.fieldShapesByType).length" class="content-shape-list"><div v-for="(shape, objectType) in selectedModule.contentContract.fieldShapesByType" :key="objectType"><strong>{{ objectType }}</strong><code>{{ shape }}</code></div></div><div v-if="Object.keys(selectedModule.contentContract.itemFieldsByType).length" class="nested-contract"><h3>条目内部必填</h3><div class="contract-decision-grid"><div v-for="(fields, objectType) in selectedModule.contentContract.itemFieldsByType" :key="objectType"><h3>{{ objectType }}</h3><div class="rule-type-list"><code v-for="field in fields" :key="field">{{ field }}</code></div></div></div></div><div class="content-threshold-list"><div v-for="(minimum, objectType) in selectedModule.contentContract.minimumContentCharsByType" :key="objectType"><strong>{{ objectType }}</strong><span>内容量参考 {{ minimum }} 字符</span><small v-if="selectedModule.contentContract.allowEmptyFieldsByType[objectType]?.length">可空：{{ selectedModule.contentContract.allowEmptyFieldsByType[objectType].join('、') }}</small></div></div><p class="contract-minimum">仅有 summary 不通过质量校验；可空字段只对对应对象合同生效。</p></section><section><h3>本模块可采用的对象合同</h3><div class="rule-type-list"><code v-for="item in selectedModule.objectTypes" :key="item">{{ item }}</code></div></section><section class="contract-example"><div><h3>正例</h3><p>{{ selectedModule.contentContract.positiveExample }}</p></div><div><h3>反例</h3><p>{{ selectedModule.contentContract.negativeExample }}</p></div></section><section><h3>适用资料与使用方</h3><p>{{ selectedModule.sources.join('、') }}</p><p class="contract-consumers">用于：{{ selectedModule.consumers.join('、') }}</p></section></div></article>
        </div>
      </section>

      <section v-else class="run-evidence-view">
        <div v-if="identification" class="evidence-layout">
          <aside class="call-list"><div><p>PROCESS & CALLS</p><h2>处理流程</h2></div><section v-for="stage in identification.processingStages" :key="stage.key" class="evidence-stage"><header><span :class="stage.actor">{{ stage.actor === 'model' ? '模型' : '代码' }}</span><div><strong>{{ stage.name }}</strong><small>{{ formatDuration(stage.durationMs) }}</small></div></header><p>{{ stage.detail }}</p><button v-for="item in callsForStage(stage)" :key="item.call.callId || `${item.call.attempt}-${item.call.purpose}`" type="button" :class="{ active: selectedCallIndex === item.index }" @click="selectedCallIndex = item.index"><span>{{ item.call.callId || item.index + 1 }}</span><div><strong>{{ callPurposeLabel(item.call.purpose) }}</strong><small>{{ item.call.segment || '全文' }} · {{ item.call.promptTokens + item.call.completionTokens }} tokens</small></div></button></section><section v-if="repairCalls.length" class="evidence-stage"><header><span class="model">模型</span><div><strong>结构修复与重试</strong><small>按失败触发</small></div></header><button v-for="item in repairCalls" :key="item.call.callId || item.index" type="button" :class="{ active: selectedCallIndex === item.index }" @click="selectedCallIndex = item.index"><span>{{ item.call.callId || item.index + 1 }}</span><div><strong>{{ callPurposeLabel(item.call.purpose) }}</strong><small>{{ item.call.retryOf ? `重试 ${item.call.retryOf}` : '定点修复' }}</small></div></button></section></aside>
          <article v-if="selectedCall" class="prompt-review"><header><div><p>{{ identification.model }} · {{ selectedCall.callId || `调用 ${selectedCallIndex + 1}` }}</p><h2>{{ callPurposeLabel(selectedCall.purpose) }}</h2></div><span>{{ selectedCall.status === 'completed' ? '完成' : '失败' }}</span></header><div class="call-facts"><span>输入 {{ selectedCall.promptTokens }} tokens</span><span>输出 {{ selectedCall.completionTokens }} tokens</span><span>耗时 {{ formatDuration(selectedCall.durationMs) }}</span><span>范围 {{ selectedCall.segment || '全文' }}</span><span v-if="selectedCall.retryOf">重试自 {{ selectedCall.retryOf }}</span><span v-if="selectedCall.finishReason">结束原因 {{ selectedCall.finishReason }}</span></div><p class="call-flow">结果去向：{{ callResultFlow(selectedCall.purpose) }}</p><div class="prompt-sections"><section><h3>模型任务与规则</h3><pre>{{ selectedCall.systemPrompt || '旧运行未记录系统规则。' }}</pre></section><section><h3>本次输入范围</h3><pre>{{ selectedCall.userPrompt || '旧运行未记录资料输入。' }}</pre></section><section><h3>模型原始输出</h3><pre>{{ prettyJson(selectedCall.rawOutput || selectedCall.error || '') }}</pre></section></div></article>
          <aside class="run-summary"><section><span>规则版本</span><strong>{{ identification.catalogVersion }}</strong></section><section><span>提示词版本</span><strong>{{ identification.promptVersion }}</strong></section><section><span>输出合同</span><strong>{{ identification.schemaVersion }}</strong></section><section><span>中间主张</span><strong>{{ identification.atomicClaims.length }}条有效 / {{ identification.rejectedAtomicClaims.length }}条拒绝</strong></section><section v-if="identification.qualityReport"><span>Gold 对齐</span><strong>{{ identification.qualityReport.groupsMet }}/{{ identification.qualityReport.groupCount }}组 · 召回代理 {{ Math.round(identification.qualityReport.objectRecallProxy * 100) }}%</strong></section><section><span>对象粒度</span><strong>{{ identification.granularityMetrics?.objectCount ?? '-' }} 个对象计划<br>平均 {{ identification.granularityMetrics?.averageClaimsPerObject ?? '-' }} 条证据主张/对象 · 同来源拆分 {{ identification.granularityMetrics?.sourceAnchorsSplitAcrossObjects ?? '-' }}</strong></section><section><span>模型参数</span><strong>temperature {{ identification.modelConfiguration?.temperature ?? '-' }}<br>max tokens {{ identification.modelConfiguration?.maxOutputTokens ?? '-' }}</strong></section><section><span>执行参数</span><strong>超时 {{ identification.modelConfiguration?.timeoutSeconds ?? '-' }}s · 重试 {{ identification.modelConfiguration?.maxRetries ?? '-' }}次<br>分段 {{ identification.modelConfiguration?.documentMaxChars ?? '-' }}字 · 并发 {{ identification.modelConfiguration?.maxConcurrency ?? '-' }}</strong></section><section><span>耗时</span><strong>{{ formatDuration(identification.durationMs) }}</strong></section><section><span>识别结果</span><strong>{{ identification.candidates.length }}项候选 / {{ identification.rejectedCandidates.length }}项拒绝</strong></section></aside>
        </div>
        <div v-else class="build-empty"><IconCode size="30" /><h2>还没有运行证据</h2><p>完成一次知识构建后，可查看模型输入、规则、资料内容和原始响应。</p><button type="button" @click="activeView = 'build'">返回构建流程</button></div>
      </section>
    </section>
  </main>
</template>
