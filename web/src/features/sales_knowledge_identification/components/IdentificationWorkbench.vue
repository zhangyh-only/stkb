<script setup lang="ts">
import {
  IconActivityHeartbeat,
  IconAlertCircle,
  IconBook2,
  IconBrain,
  IconDatabase,
  IconFileText,
  IconFlask2,
  IconGitBranch,
  IconPlayerPlay,
  IconRefresh,
  IconSearch,
  IconShieldCheck,
  IconStack2,
  IconTable,
} from '@tabler/icons-vue'
import { computed, onMounted, ref, type Component } from 'vue'

import {
  apiBaseUrl,
  getDocumentPackage,
  getIdentificationCatalog,
  getIdentificationEvaluation,
  getIdentificationRun,
  IdentificationApiError,
  listIdentificationRuns,
  listSourceMaterials,
  runIdentification,
} from '../api'
import {
  COVERAGE_LABELS,
  coverageCount,
  formatDuration,
  prettyJson,
  type CandidateKnowledgeObject,
  type CoverageStatus,
  type DocumentPackage,
  type IdentificationResult,
  type IdentificationCatalog,
  type KnowledgeModule,
  type ModelCallTrace,
  type ProcessingStage,
  type SourceAnchor,
  type SourceMaterial,
} from '../types'

type WorkbenchTab = 'overview' | 'source' | 'rules' | 'candidates' | 'coverage' | 'evaluation' | 'traces'
type ProposalView = 'accepted' | 'rejected' | 'signals' | 'normalizations'

const tabItems: { key: WorkbenchTab; label: string; icon: Component }[] = [
  { key: 'overview', label: '运行总览', icon: IconActivityHeartbeat },
  { key: 'source', label: '输入资料', icon: IconFileText },
  { key: 'rules', label: '规则审阅', icon: IconBook2 },
  { key: 'candidates', label: '对象提议', icon: IconStack2 },
  { key: 'coverage', label: '模块覆盖', icon: IconTable },
  { key: 'evaluation', label: '代理评估', icon: IconShieldCheck },
  { key: 'traces', label: '调用与阶段', icon: IconGitBranch },
]

const selectedMaterialId = ref('')
const sourceMaterials = ref<SourceMaterial[]>([])
const documentPackage = ref<DocumentPackage | null>(null)
const result = ref<IdentificationResult | null>(null)
const selectedCandidateId = ref<string | null>(null)
const selectedEvidenceRef = ref<string | null>(null)
const proposalView = ref<ProposalView>('accepted')
const activeTab = ref<WorkbenchTab>('overview')
const packageLoading = ref(false)
const runLoading = ref(false)
const error = ref('')
const errorEndpoint = ref('')
const apiState = ref<'checking' | 'ready' | 'error'>('checking')
const packageNotice = ref('')
const runNotice = ref('')
const lastRunId = ref('')
const runHistory = ref<IdentificationResult[]>([])
const proxyEvaluation = ref('')
const catalogModules = ref<KnowledgeModule[]>([])
const catalogInfo = ref<Omit<IdentificationCatalog, 'modules'> | null>(null)
const selectedRuleDomainCode = ref('D1')
const selectedRuleModuleCode = ref('D1.1')

const selectedCandidate = computed<CandidateKnowledgeObject | null>(() => {
  const candidates = result.value?.candidates ?? []
  return candidates.find((candidate) => candidate.candidateId === selectedCandidateId.value) ?? candidates[0] ?? null
})

const selectedEvidence = computed(() =>
  selectedCandidate.value?.evidence.find((reference) => reference === selectedEvidenceRef.value)
    ?? selectedCandidate.value?.evidence[0]
    ?? null,
)

const groupedModules = computed(() => {
  const domains = new Map<string, { domain: string; label: string; modules: KnowledgeModule[] }>()
  for (const module of catalogModules.value) {
    const group = domains.get(module.domain) ?? {
      domain: module.domain,
      label: module.domainName,
      modules: [],
    }
    group.modules.push(module)
    domains.set(module.domain, group)
  }
  return [...domains.values()]
})

const moduleCount = computed(() => catalogModules.value.length)

const selectedRuleDomain = computed(() =>
  catalogInfo.value?.domains.find((domain) => domain.code === selectedRuleDomainCode.value) ?? null,
)

const selectedRuleModules = computed(() =>
  catalogModules.value.filter((module) => module.domain === selectedRuleDomainCode.value),
)

const selectedRuleModule = computed(() =>
  selectedRuleModules.value.find((module) => module.code === selectedRuleModuleCode.value)
    ?? selectedRuleModules.value[0]
    ?? null,
)

const coverageSummary = computed(() => ({
  hit: coverageCount(result.value?.coverageByModule, 'hit'),
  weakSignal: coverageCount(result.value?.coverageByModule, 'weak_signal'),
  unresolved: coverageCount(result.value?.coverageByModule, 'unresolved'),
  notFound: coverageCount(result.value?.coverageByModule, 'not_found'),
}))

const parsedRawOutput = computed(() => {
  if (!result.value?.rawModelOutput) return ''
  return prettyJson(result.value.rawModelOutput)
})

const runStability = computed(() => {
  if (runHistory.value.length < 2) return null
  const candidateSets = runHistory.value.map(
    (run) => new Set(run.candidates.map((candidate) => `${candidate.module}:${candidate.objectType}`)),
  )
  const stableKnowledgeTypes = [...candidateSets[0]].filter((knowledgeType) =>
    candidateSets.every((candidateSet) => candidateSet.has(knowledgeType)),
  )
  const allKnowledgeTypes = new Set(candidateSets.flatMap((candidateSet) => [...candidateSet]))
  return {
    runCount: runHistory.value.length,
    stableCount: stableKnowledgeTypes.length,
    changedCount: allKnowledgeTypes.size - stableKnowledgeTypes.length,
  }
})

function coverageStatus(moduleCode: string): CoverageStatus {
  return result.value?.coverageByModule[moduleCode] ?? 'not_found'
}

function anchorFor(reference: string): SourceAnchor | undefined {
  return documentPackage.value?.anchors.find((anchor) => anchor.anchorId === reference)
}

function evidenceLabel(reference: string): string {
  const anchor = anchorFor(reference)
  if (!anchor) return reference
  return anchor.page ? `${reference} · 第 ${anchor.page} 页` : `${reference} · ${anchor.kind}`
}

function evidenceExcerpt(reference: string): string {
  const markdown = documentPackage.value?.fullMarkdown ?? ''
  const marker = `<!-- source-anchor: ${reference} -->`
  const markerIndex = markdown.indexOf(marker)
  if (markerIndex < 0) return ''
  const contentStart = markerIndex + marker.length
  const nextAnchorIndex = markdown.indexOf('<!-- source-anchor:', contentStart)
  const contentEnd = nextAnchorIndex < 0 ? markdown.length : nextAnchorIndex
  return markdown.slice(contentStart, contentEnd).trim().slice(0, 900)
}

function candidateSummary(candidate: CandidateKnowledgeObject): string {
  if (candidate.title) return candidate.title
  const values = Object.values(candidate.content)
  const firstText = values.find((value) => typeof value === 'string')
  return typeof firstText === 'string' ? firstText : `${candidate.objectType} · ${candidate.module}`
}

function stageClass(stage: ProcessingStage): string {
  return stage.status === 'completed' ? 'stage-complete' : 'stage-failed'
}

function callClass(call: ModelCallTrace): string {
  return call.status === 'completed' ? 'call-complete' : 'call-failed'
}

function loadErrorMessage(reason: unknown): string {
  if (reason instanceof IdentificationApiError) {
    errorEndpoint.value = reason.endpoint
    if (reason.status === 404) {
      return `请求的资料不存在（${reason.message}）。请重新选择资料，或重启前端以加载最新 API 配置。`
    }
    return reason.message
  }
  if (reason instanceof Error) return reason.message
  return '请求失败，请检查识别服务和原始资料登记。'
}

function clearError(): void {
  error.value = ''
  errorEndpoint.value = ''
}

async function probeApi(): Promise<void> {
  apiState.value = 'checking'
  try {
    const [catalog, materials] = await Promise.all([
      getIdentificationCatalog(),
      listSourceMaterials(),
    ])
    catalogModules.value = catalog.modules
    catalogInfo.value = {
      version: catalog.version,
      fingerprint: catalog.fingerprint,
      status: catalog.status,
      source: catalog.source,
      scopeDefinitions: catalog.scopeDefinitions,
      domains: catalog.domains,
    }
    sourceMaterials.value = materials
    apiState.value = 'ready'
  } catch (reason) {
    apiState.value = 'error'
    error.value = loadErrorMessage(reason)
  }
}

async function loadPackage(): Promise<void> {
  const id = selectedMaterialId.value
  if (!id) {
    error.value = '请先选择一份原始资料。'
    return
  }
  packageLoading.value = true
  clearError()
  packageNotice.value = ''
  runNotice.value = ''
  try {
    const [nextPackage, catalog] = await Promise.all([
      getDocumentPackage(id),
      getIdentificationCatalog(),
    ])
    documentPackage.value = nextPackage
    result.value = null
    runHistory.value = []
    lastRunId.value = ''
    selectedCandidateId.value = null
    selectedEvidenceRef.value = null
    proposalView.value = 'accepted'
    proxyEvaluation.value = ''
    catalogModules.value = catalog.modules
    catalogInfo.value = {
      version: catalog.version,
      fingerprint: catalog.fingerprint,
      status: catalog.status,
      source: catalog.source,
      scopeDefinitions: catalog.scopeDefinitions,
      domains: catalog.domains,
    }
    apiState.value = 'ready'
    activeTab.value = 'overview'
    packageNotice.value = `已读取《${nextPackage.sourceFileName}》。历史运行结果未自动载入，请执行识别查看本轮结果。`
  } catch (reason) {
    error.value = loadErrorMessage(reason)
  } finally {
    packageLoading.value = false
  }
}

async function executeRun(): Promise<void> {
  const id = selectedMaterialId.value
  if (!id) {
    error.value = '请先选择一份原始资料。'
    return
  }
  runLoading.value = true
  clearError()
  runNotice.value = ''
  try {
    if (!documentPackage.value || documentPackage.value.documentPackageId !== id) {
      documentPackage.value = await getDocumentPackage(id)
    }
    const nextResult = await runIdentification(id)
    const [history, evaluation] = await Promise.all([
      listIdentificationRuns(id).catch(() => []),
      getIdentificationEvaluation(id).catch(() => null),
    ])
    result.value = nextResult
    runHistory.value = [
      ...history.filter((run) => run.runId !== nextResult.runId),
      nextResult,
    ].slice(-5)
    proxyEvaluation.value = evaluation?.markdown ?? ''
    lastRunId.value = nextResult.runId
    selectedCandidateId.value = nextResult.candidates[0]?.candidateId ?? null
    selectedEvidenceRef.value = nextResult.candidates[0]?.evidence[0] ?? null
    proposalView.value = 'accepted'
    activeTab.value = nextResult.status === 'completed' ? 'candidates' : 'traces'
    runNotice.value = nextResult.status === 'completed'
      ? `运行 ${nextResult.runId} 已完成，模型调用 ${nextResult.callCount} 次。`
      : `运行 ${nextResult.runId} 失败，失败轨迹已写入运行账本。`
  } catch (reason) {
    error.value = loadErrorMessage(reason)
  } finally {
    runLoading.value = false
  }
}

async function reloadRun(): Promise<void> {
  if (!lastRunId.value) return
  runLoading.value = true
  clearError()
  try {
    result.value = await getIdentificationRun(lastRunId.value)
    runHistory.value = [
      ...runHistory.value.filter((run) => run.runId !== result.value?.runId),
      result.value,
    ].slice(-5)
    selectedCandidateId.value = result.value.candidates[0]?.candidateId ?? null
    selectedEvidenceRef.value = result.value.candidates[0]?.evidence[0] ?? null
    runNotice.value = `已回读运行 ${lastRunId.value}`
  } catch (reason) {
    error.value = loadErrorMessage(reason)
  } finally {
    runLoading.value = false
  }
}

function selectCandidate(candidateId: string): void {
  selectedCandidateId.value = candidateId
  selectedEvidenceRef.value = result.value?.candidates.find(
    (candidate) => candidate.candidateId === candidateId,
  )?.evidence[0] ?? null
  proposalView.value = 'accepted'
  activeTab.value = 'candidates'
}

function selectRuleDomain(domainCode: string): void {
  selectedRuleDomainCode.value = domainCode
  selectedRuleModuleCode.value = catalogModules.value.find(
    (module) => module.domain === domainCode,
  )?.code ?? ''
}

onMounted(() => {
  void probeApi()
})
</script>

<template>
  <main class="lab-layout">
    <aside class="lab-sidebar" aria-label="STKB 能力验证导航">
      <div class="brand-lockup">
        <span class="brand-mark">ST</span>
        <div><strong>STKB</strong><span>能力验证实验室</span></div>
      </div>

      <nav class="capability-nav">
        <a class="nav-item muted-item" href="#" aria-disabled="true"><IconDatabase size="16" stroke="1.8" /> 三形态投影</a>
        <a class="nav-item active" href="#"><IconBrain size="16" stroke="1.8" /> 销售知识识别</a>
        <a class="nav-item muted-item" href="#" aria-disabled="true"><IconFlask2 size="16" stroke="1.8" /> 增量归并验证</a>
      </nav>

      <div class="sidebar-boundary">
        <span>验证边界</span>
        <strong>只形成对象提议</strong>
        <p>销售知识识别形成候选知识，不直接生成正式 KnowledgeObject；正式对象需要后续实体归一与知识归并。</p>
      </div>
    </aside>

    <section class="workbench-shell">
      <div class="app-bar">
        <div class="breadcrumb-line">
          <span>STKB</span>
          <span>方案验证</span>
          <strong>销售知识识别</strong>
        </div>
      <div class="service-state" :class="`service-${apiState}`">
        <span class="service-indicator"></span>
        <span>{{ apiState === 'ready' ? '识别服务已连接' : apiState === 'error' ? '识别服务异常' : '正在检查服务' }}</span>
        <code>{{ apiBaseUrl }}</code>
      </div>
      </div>

    <header class="workbench-header">
      <div>
        <h1>销售知识识别调试台</h1>
        <p class="lede">
          选择项目内已完成代理解析的原始资料，调用真实模型形成候选知识对象提议，并复核对象边界、来源证据和 D1-D5 归属。
        </p>
      </div>
      <div class="header-facts" aria-label="当前验证范围">
        <div><span>输入</span><strong>原始资料</strong></div>
        <div><span>调用</span><strong>真实模型</strong></div>
        <div><span>输出</span><strong>候选知识</strong></div>
      </div>
    </header>

    <section class="control-panel panel">
      <div class="control-heading">
        <div>
          <p class="panel-label">运行控制</p>
          <h2>选择资料并启动识别</h2>
        </div>
        <span class="run-mode"><IconBrain size="14" stroke="2" /> 真实模型</span>
      </div>
      <div class="control-row">
        <label class="package-input">
          <span>原始资料 <small>已绑定代理解析结果</small></span>
          <select v-model="selectedMaterialId">
            <option value="">请选择项目内原始资料</option>
            <option
              v-for="material in sourceMaterials"
              :key="material.documentPackageId"
              :value="material.documentPackageId"
              :disabled="material.status !== 'available'"
            >
              {{ material.sourceFileName }}{{ material.status !== 'available' ? '（不可用）' : '' }}
            </option>
          </select>
        </label>
        <button class="button button-quiet" type="button" :disabled="packageLoading || runLoading || !selectedMaterialId" @click="loadPackage">
          <IconSearch size="16" stroke="2" />
          {{ packageLoading ? '读取中…' : '读取资料' }}
        </button>
        <button class="button button-primary" type="button" :disabled="runLoading || packageLoading || !selectedMaterialId" @click="executeRun">
          <IconPlayerPlay v-if="!runLoading" size="16" stroke="2" />
          <span v-else class="button-loader"></span>
          {{ runLoading ? '模型识别运行中…' : '执行真实模型识别' }}
        </button>
      </div>
      <div class="control-meta">
        <span>API <code>{{ apiBaseUrl }}</code></span>
        <span>{{ sourceMaterials.length }} 份资料已登记</span>
        <span v-if="packageNotice" class="success-text">{{ packageNotice }}</span>
        <span v-if="runNotice" class="success-text">{{ runNotice }}</span>
      </div>
    </section>

    <div v-if="error" class="alert alert-error" role="alert">
      <span class="alert-icon"><IconAlertCircle size="18" stroke="2" /></span>
      <div class="alert-copy">
        <strong>当前请求未完成</strong>
        <span>{{ error }}</span>
        <code v-if="errorEndpoint">{{ errorEndpoint }}</code>
      </div>
      <button type="button" class="alert-action" @click="probeApi"><IconRefresh size="14" stroke="2" />重新检查服务</button>
    </div>

    <section v-if="documentPackage" class="package-strip panel">
      <div class="package-identity">
        <span class="status-dot" :class="documentPackage.status"></span>
        <div>
          <p class="eyebrow-small">DOCUMENT PACKAGE</p>
          <h2>{{ documentPackage.sourceFileName }}</h2>
          <p class="muted">{{ documentPackage.sourceFilePath }}</p>
        </div>
      </div>
      <div class="package-facts">
        <div><span>状态</span><strong>{{ documentPackage.status === 'available' ? '可用' : '不可用' }}</strong></div>
        <div><span>解析方式</span><strong>{{ documentPackage.processingMethod === 'agent_assisted' ? '代理解析' : '能力解析' }}</strong></div>
        <div><span>来源锚点</span><strong>{{ documentPackage.anchors.length }} 个</strong></div>
        <div><span>质量问题</span><strong :class="{ 'warning-text': documentPackage.qualityIssues.length }">{{ documentPackage.qualityIssues.length }} 项</strong></div>
      </div>
    </section>

    <nav class="workbench-tabs" aria-label="调试视图">
      <button v-for="tab in tabItems" :key="tab.key" type="button" :class="{ active: activeTab === tab.key }" @click="activeTab = tab.key">
        <component :is="tab.icon" size="16" stroke="1.8" />
        {{ tab.key === 'coverage' ? `${moduleCount} 模块覆盖` : tab.label }}
        <span v-if="tab.key === 'candidates' && result" class="tab-count">{{ result.candidates.length }}</span>
        <span v-if="tab.key === 'coverage' && result" class="tab-count">{{ coverageSummary.hit }}/{{ moduleCount }}</span>
      </button>
    </nav>

    <template v-if="documentPackage || activeTab === 'rules'">
      <section v-if="activeTab === 'overview'" class="view-grid">
        <div class="main-column">
          <div v-if="result" class="metrics-grid">
            <article class="metric-card metric-accent">
              <span>对象提议</span>
              <strong>{{ result.candidates.length }}</strong>
              <small>通过程序合同与证据校验</small>
            </article>
            <article class="metric-card">
              <span>模型调用</span>
              <strong>{{ result.callCount }}</strong>
              <small>{{ result.promptTokens + result.completionTokens }} tokens</small>
            </article>
            <article class="metric-card">
              <span>总耗时</span>
              <strong>{{ formatDuration(result.durationMs) }}</strong>
              <small>{{ result.provider }} / {{ result.model }}</small>
            </article>
            <article class="metric-card">
              <span>证据异常</span>
              <strong :class="{ 'warning-text': result.rejectedCandidates.length }">{{ result.rejectedCandidates.length }}</strong>
              <small>拒绝候选，不进入下游</small>
            </article>
          </div>
          <div v-else class="empty-state panel">
            <span class="empty-index">02</span>
            <div>
              <h2>资料已就绪，等待一次真实识别</h2>
              <p>点击“执行真实模型识别”后，后端将把全文、销售知识规则包和证据规则组装为模型请求。没有真实运行结果前，页面不会生成对象提议或模拟调用数据。</p>
            </div>
          </div>

          <div v-if="result" class="panel section-panel">
            <div class="section-title-row">
              <div><p class="eyebrow-small">PROCESSING TRACE</p><h2>识别处理阶段</h2></div>
              <button class="link-button" type="button" :disabled="runLoading" @click="reloadRun"><IconRefresh size="14" stroke="2" />回读运行</button>
            </div>
            <div class="stage-list">
              <div v-for="stage in result.processingStages" :key="stage.key" class="stage-item">
                <span class="stage-marker" :class="stageClass(stage)"></span>
                <div class="stage-copy"><strong>{{ stage.name }}</strong><span>{{ stage.detail }}</span></div>
                <time>{{ formatDuration(stage.durationMs) }}</time>
              </div>
            </div>
          </div>

          <div v-if="result" class="panel section-panel">
            <div class="section-title-row">
              <div><p class="eyebrow-small">RESULT BOUNDARY</p><h2>本节点产出边界</h2></div>
              <span class="pill" :class="result.status === 'completed' ? 'pill-green' : 'pill-red'">{{ result.status === 'completed' ? '识别完成' : '识别失败' }}</span>
            </div>
            <div class="boundary-grid">
              <div><span>PostgreSQL 运行账本</span><strong>{{ result.storageImpact.postgresRunRecords }} 条</strong><em class="is-written">本节点写入声明</em></div>
              <div><span>正式知识 Markdown</span><strong>{{ result.storageImpact.formalKnowledgeFiles }} 条</strong><em>本节点不写入</em></div>
              <div><span>pgvector</span><strong>{{ result.storageImpact.pgvectorRecords }} 条</strong><em>本节点不写入</em></div>
              <div><span>Neo4j</span><strong>{{ result.storageImpact.neo4jNodes }} 节点 / {{ result.storageImpact.neo4jRelationships }} 关系</strong><em>本节点不写入；实测见代理评估</em></div>
            </div>
          </div>
        </div>

        <aside class="side-column">
          <div class="panel side-panel">
            <div class="section-title-row"><div><p class="eyebrow-small">MODEL CONTRACT</p><h2>运行配置</h2></div><span class="pill">真实调用</span></div>
            <dl class="detail-list">
              <div><dt>Provider</dt><dd>{{ result?.provider ?? '等待运行' }}</dd></div>
              <div><dt>Model</dt><dd>{{ result?.model ?? '配置后显示' }}</dd></div>
              <div><dt>Prompt</dt><dd>{{ result?.promptVersion ?? '-' }}</dd></div>
              <div><dt>Schema</dt><dd>{{ result?.schemaVersion ?? '-' }}</dd></div>
              <div><dt>Catalog</dt><dd>{{ result?.catalogVersion ?? '-' }}</dd></div>
              <div><dt>分段阈值</dt><dd>{{ result?.modelConfiguration?.documentMaxChars ?? '旧运行未记录' }}</dd></div>
              <div><dt>并发上限</dt><dd>{{ result?.modelConfiguration?.maxConcurrency ?? '旧运行未记录' }}</dd></div>
              <div><dt>配置指纹</dt><dd class="mono">{{ result?.modelConfiguration?.fingerprint ?? '旧运行未记录' }}</dd></div>
              <div><dt>Run ID</dt><dd class="mono">{{ result?.runId ?? '-' }}</dd></div>
            </dl>
          </div>
          <div class="panel side-panel">
            <div class="section-title-row"><div><p class="eyebrow-small">SIGNAL SUMMARY</p><h2>识别信号</h2></div></div>
            <div class="signal-list">
              <div><span class="signal-bar signal-hit"></span><span>有效候选</span><strong>{{ coverageSummary.hit }}</strong></div>
              <div><span class="signal-bar signal-weak"></span><span>弱线索</span><strong>{{ result?.weakSignals.length ?? 0 }}</strong></div>
              <div><span class="signal-bar signal-unresolved"></span><span>未决项</span><strong>{{ result?.unresolvedItems.length ?? 0 }}</strong></div>
              <div><span class="signal-bar signal-rejected"></span><span>拒绝候选</span><strong>{{ result?.rejectedCandidates.length ?? 0 }}</strong></div>
            </div>
          </div>
          <div class="proxy-note">
            <span class="note-icon">◎</span>
            <div><strong>评估边界</strong><p>模型原始输出与程序校验结果保持只读。代理评估在本节点之外独立记录，不修改模型结果。</p></div>
          </div>
          <div v-if="runStability" class="panel side-panel stability-panel">
            <div class="section-title-row"><div><p class="eyebrow-small">RUN STABILITY</p><h2>重复运行对比</h2></div><span class="pill">本页 {{ runStability.runCount }} 次</span></div>
            <div class="stability-metrics"><div><strong>{{ runStability.stableCount }}</strong><span>稳定知识类型</span></div><div><strong>{{ runStability.changedCount }}</strong><span>发生变化</span></div></div>
            <p class="stability-note">从 PostgreSQL 回读最近运行，按模块与对象类型比较；该视图用于发现漂移，不替代代理评估。</p>
          </div>
        </aside>
      </section>

      <section v-else-if="activeTab === 'source' && documentPackage" class="source-view">
        <div class="source-meta-grid">
          <div class="panel source-card"><p class="eyebrow-small">SOURCE FILE</p><h2>{{ documentPackage.sourceFileName }}</h2><dl class="detail-list"><div><dt>项目内原件</dt><dd class="mono">{{ documentPackage.sourceFilePath }}</dd></div><div><dt>原件 SHA-256</dt><dd class="mono hash-value">{{ documentPackage.sourceSha256 }}</dd></div><div><dt>全文路径</dt><dd class="mono">{{ documentPackage.fullMarkdownPath }}</dd></div><div><dt>全文 SHA-256</dt><dd class="mono hash-value">{{ documentPackage.fullMarkdownSha256 }}</dd></div></dl></div>
          <div class="panel source-card"><p class="eyebrow-small">QUALITY NOTES</p><h2>{{ documentPackage.qualityIssues.length ? '需要留意' : '暂无质量问题' }}</h2><ul v-if="documentPackage.qualityIssues.length" class="issue-list"><li v-for="issue in documentPackage.qualityIssues" :key="issue">{{ issue }}</li></ul><p v-else class="muted">代理解析结果未登记质量问题。</p><div class="method-stamp">{{ documentPackage.processingMethod === 'agent_assisted' ? 'AGENT ASSISTED' : 'CAPABILITY OUTPUT' }}</div></div>
        </div>
        <div class="panel source-document"><div class="section-title-row"><div><p class="eyebrow-small">FULL MARKDOWN</p><h2>可定位的全文输入</h2></div><span class="pill">只读</span></div><pre class="markdown-viewer">{{ documentPackage.fullMarkdown }}</pre></div>
        <div class="panel source-anchors"><div class="section-title-row"><div><p class="eyebrow-small">EVIDENCE ANCHORS</p><h2>来源锚点</h2></div><span class="muted">{{ documentPackage.anchors.length }} 个</span></div><div class="anchor-grid"><div v-for="anchor in documentPackage.anchors" :key="anchor.anchorId" class="anchor-item"><code>{{ anchor.anchorId }}</code><span>{{ anchor.kind }}<template v-if="anchor.page"> · 第 {{ anchor.page }} 页</template></span></div></div></div>
      </section>

      <section v-else-if="activeTab === 'rules'" class="rules-view">
        <div class="panel rules-meta">
          <div>
            <p class="eyebrow-small">SALES KNOWLEDGE MODEL</p>
            <h2>销售知识模型规则包</h2>
            <p>左侧选销售域，中间选知识模块，右侧审阅当前生效的完整规则。规则文件是模型识别和程序校验共同使用的事实源。</p>
          </div>
          <dl class="rules-version">
            <div><dt>版本</dt><dd>{{ catalogInfo?.version ?? '-' }}</dd></div>
            <div><dt>状态</dt><dd>{{ catalogInfo?.status === 'sample_validation' ? '样本验证中' : '-' }}</dd></div>
            <div><dt>来源</dt><dd class="mono">{{ catalogInfo?.source ?? '-' }}</dd></div>
            <div><dt>指纹</dt><dd class="mono hash-value">{{ catalogInfo?.fingerprint ?? '-' }}</dd></div>
          </dl>
          <div class="rule-scope-legend">
            <p><b>核心范围</b>{{ catalogInfo?.scopeDefinitions.core }}</p>
            <p><b>可选范围</b>{{ catalogInfo?.scopeDefinitions.optional }}</p>
          </div>
        </div>

        <div class="panel rules-workspace">
          <nav class="rule-domain-nav" aria-label="销售知识域">
            <p>销售知识域</p>
            <button
              v-for="domain in catalogInfo?.domains ?? []"
              :key="domain.code"
              type="button"
              :class="{ active: selectedRuleDomainCode === domain.code }"
              @click="selectRuleDomain(domain.code)"
            >
              <span>{{ domain.code }}</span>
              <div><strong>{{ domain.name }}</strong><em>{{ domain.question }}</em></div>
            </button>
          </nav>

          <div class="rule-module-nav">
            <div class="rule-pane-head">
              <div><span>{{ selectedRuleDomain?.code }}</span><strong>{{ selectedRuleDomain?.name }}</strong></div>
              <small>{{ selectedRuleModules.length }} 个模块</small>
            </div>
            <div class="rule-module-options">
              <button
                v-for="module in selectedRuleModules"
                :key="module.code"
                type="button"
                :class="{ active: selectedRuleModule?.code === module.code }"
                @click="selectedRuleModuleCode = module.code"
              >
                <code>{{ module.code }}</code>
                <div><strong>{{ module.name }}</strong><span>{{ module.scope === 'optional' ? '可选范围' : '核心范围' }}</span></div>
              </button>
            </div>
          </div>

          <article v-if="selectedRuleDomain && selectedRuleModule" class="rule-detail-pane">
            <header>
              <div><span>{{ selectedRuleModule.code }}</span><h2>{{ selectedRuleModule.name }}</h2></div>
              <em>{{ selectedRuleModule.scope === 'optional' ? '可选范围模块' : '核心范围模块' }}</em>
            </header>

            <section class="domain-context">
              <div><span>{{ selectedRuleDomain.question }}</span><strong>{{ selectedRuleDomain.meaning }}</strong></div>
              <p><b>域边界</b>{{ selectedRuleDomain.boundary }}</p>
            </section>

            <div class="rule-detail-scroll">
              <section><h3>模块业务含义</h3><p>{{ selectedRuleModule.meaning }}</p></section>
              <section class="rule-emphasis"><h3>对象边界与分类裁决</h3><p>{{ selectedRuleModule.boundary }}</p></section>
              <section><h3>允许形成的对象类型</h3><div class="rule-chips"><code v-for="item in selectedRuleModule.objectTypes" :key="item">{{ item }}</code></div></section>
              <section><h3>核心对象表达</h3><p>{{ selectedRuleModule.coreObjects.join('、') }}</p></section>
              <section><h3>典型来源资料</h3><p>{{ selectedRuleModule.sources.join('、') }}</p></section>
              <section><h3>明确消费方</h3><p>{{ selectedRuleModule.consumers.join('、') }}</p></section>
            </div>
          </article>
        </div>
      </section>

      <section v-else-if="activeTab === 'candidates'" class="candidate-view">
        <template v-if="result">
          <div class="object-formation-strip panel">
            <div class="formation-step completed"><span>01</span><div><strong>候选知识对象</strong><p>{{ result.candidates.length }} 项，已完成对象边界、分类、合同与证据校验</p></div></div>
            <div class="formation-arrow">→</div>
            <div class="formation-step pending"><span>02</span><div><strong>业务实体归一</strong><p>本节点未执行，实体提及还没有正式实体 ID</p></div></div>
            <div class="formation-arrow">→</div>
            <div class="formation-step pending"><span>03</span><div><strong>正式 KnowledgeObject</strong><p>当前 0 项；需经过跨资料归并和正式写入</p></div></div>
          </div>
          <nav class="proposal-result-nav panel" aria-label="识别结果类型">
            <button type="button" :class="{ active: proposalView === 'accepted' }" @click="proposalView = 'accepted'"><span>候选对象</span><strong>{{ result.candidates.length }}</strong></button>
            <button type="button" :class="{ active: proposalView === 'rejected' }" @click="proposalView = 'rejected'"><span>拒绝项</span><strong>{{ result.rejectedCandidates.length }}</strong></button>
            <button type="button" :class="{ active: proposalView === 'signals' }" @click="proposalView = 'signals'"><span>弱线索 / 未决</span><strong>{{ result.weakSignals.length + result.unresolvedItems.length }}</strong></button>
            <button type="button" :class="{ active: proposalView === 'normalizations' }" @click="proposalView = 'normalizations'"><span>规范化</span><strong>{{ result.normalizations.length }}</strong></button>
          </nav>

          <div class="panel proposal-workspace">
            <template v-if="proposalView === 'accepted'">
              <aside class="proposal-list-pane">
                <div class="proposal-pane-head"><div><p>OBJECT PROPOSALS</p><h2>候选知识对象</h2></div><span>{{ result.candidates.length }} 项</span></div>
                <div class="proposal-list-scroll">
                  <button v-for="candidate in result.candidates" :key="candidate.candidateId" type="button" :class="{ active: selectedCandidate?.candidateId === candidate.candidateId }" @click="selectCandidate(candidate.candidateId)">
                    <span>{{ candidate.candidateId }}</span>
                    <div><strong>{{ candidateSummary(candidate) }}</strong><small>{{ candidate.module }} · {{ candidate.objectType }}</small></div>
                  </button>
                  <p v-if="!result.candidates.length" class="empty-inline">本次运行没有形成通过校验的候选知识对象。</p>
                </div>
              </aside>

              <article v-if="selectedCandidate" class="proposal-detail-pane">
                <header>
                  <div><p>{{ selectedCandidate.candidateId }} · {{ selectedCandidate.domain }} / {{ selectedCandidate.module }}</p><h2>{{ selectedCandidate.title || selectedCandidate.objectType }}</h2><span>{{ selectedCandidate.objectType }}</span></div>
                  <em>运行内候选身份</em>
                </header>
                <div class="proposal-detail-scroll">
                  <section class="proposal-contract-grid">
                    <div><h3>对象边界</h3><p>{{ selectedCandidate.objectBoundary || '旧运行未提供对象边界，需重新识别。' }}</p></div>
                    <div><h3>分类依据</h3><p>{{ selectedCandidate.classificationBasis || '旧运行未提供分类依据，需重新识别。' }}</p></div>
                  </section>
                  <section><h3>身份线索（供后续归并比较）</h3><pre class="json-viewer compact-object">{{ prettyJson(selectedCandidate.identityHints) }}</pre></section>
                  <section><h3>类型化业务内容</h3><pre class="json-viewer compact-object">{{ prettyJson(selectedCandidate.content) }}</pre></section>
                  <section>
                    <h3>来源证据</h3>
                    <div class="evidence-selector">
                      <button v-for="reference in selectedCandidate.evidence" :key="reference" type="button" :class="{ active: selectedEvidence === reference }" @click="selectedEvidenceRef = reference">{{ evidenceLabel(reference) }}</button>
                    </div>
                    <div v-if="selectedEvidence" class="single-evidence-view"><span :class="anchorFor(selectedEvidence) ? 'evidence-ok' : 'evidence-missing'">{{ anchorFor(selectedEvidence) ? '锚点有效' : '锚点缺失' }}</span><p>{{ evidenceExcerpt(selectedEvidence) || '当前锚点没有可展示的文本片段。' }}</p></div>
                  </section>
                  <section v-if="selectedCandidate.entityMentions.length"><h3>实体提及与引用角色</h3><div class="mention-list"><div v-for="mention in selectedCandidate.entityMentions" :key="mention.mentionId" class="mention-item"><strong>{{ mention.text }}</strong><span>{{ mention.proposedType }} · {{ mention.referenceRole }}</span><code>{{ mention.sourceRef }}</code></div></div></section>
                  <section v-if="selectedCandidate.relations.length"><h3>关系建议</h3><div class="relation-list"><div v-for="(relation, index) in selectedCandidate.relations" :key="`${relation.relationType}-${index}`"><span class="relation-kind">{{ relation.relationKind }}</span><strong>{{ relation.sourceRef }}</strong><span>→</span><strong>{{ relation.targetRef }}</strong><span class="muted">{{ relation.relationType }}</span></div></div></section>
                </div>
              </article>
              <div v-else class="proposal-empty">没有可展示的候选知识对象。</div>
            </template>

            <section v-else-if="proposalView === 'rejected'" class="proposal-aux-pane"><div class="proposal-pane-head"><div><p>REJECTED</p><h2>被程序拒绝的对象提议</h2></div><span>{{ result.rejectedCandidates.length }} 项</span></div><div class="proposal-aux-scroll"><details v-for="item in result.rejectedCandidates" :key="item.candidateId"><summary><strong>{{ item.candidateId }}</strong><span>{{ item.reasons.join('；') }}</span></summary><pre class="json-viewer compact-object">{{ prettyJson(item.rawCandidate) }}</pre></details><p v-if="!result.rejectedCandidates.length" class="proposal-empty">本次没有被程序拒绝的对象提议。</p></div></section>

            <section v-else-if="proposalView === 'signals'" class="proposal-aux-pane"><div class="proposal-pane-head"><div><p>WEAK / UNRESOLVED</p><h2>弱线索与未决项</h2></div><span>{{ result.weakSignals.length + result.unresolvedItems.length }} 项</span></div><div class="proposal-aux-scroll"><div v-for="(item, index) in result.weakSignals" :key="`weak-${index}`" class="proposal-signal"><span>弱线索 · {{ item.module }}</span><p>{{ item.reason }}</p><code v-for="reference in item.evidence" :key="reference">{{ reference }}</code></div><div v-for="(item, index) in result.unresolvedItems" :key="`unresolved-${index}`" class="proposal-signal unresolved"><span>未决{{ item.module ? ` · ${item.module}` : '' }}</span><p>{{ item.description }}：{{ item.reason }}</p></div><p v-if="!result.weakSignals.length && !result.unresolvedItems.length" class="proposal-empty">本次没有弱线索或未决项。</p></div></section>

            <section v-else class="proposal-aux-pane"><div class="proposal-pane-head"><div><p>DETERMINISTIC NORMALIZATION</p><h2>程序规范化记录</h2></div><span>{{ result.normalizations.length }} 项</span></div><div class="proposal-aux-scroll"><div v-for="item in result.normalizations" :key="`${item.candidateId}-${item.field}`" class="normalization-row"><strong>{{ item.candidateId }}</strong><code>{{ item.field }}：{{ item.originalValue }} → {{ item.normalizedValue }}</code><p>{{ item.reason }}</p></div><p v-if="!result.normalizations.length" class="proposal-empty">本次没有程序规范化记录。</p></div></section>
          </div>
        </template>
        <div v-else class="panel empty-state"><span class="empty-index">03</span><div><h2>运行后查看候选知识对象</h2><p>每项候选对象必须包含标题、对象边界、分类依据、身份线索、类型化内容和来源证据；正式 KnowledgeObject 仍由后续归并节点形成。</p></div></div>
      </section>

      <section v-else-if="activeTab === 'coverage'" class="coverage-view">
        <div class="panel coverage-summary"><div class="section-title-row"><div><p class="eyebrow-small">CATALOG / {{ moduleCount }} MODULES</p><h2>覆盖矩阵</h2></div><div class="coverage-legend"><span><i class="legend-dot hit"></i>已命中 {{ coverageSummary.hit }}</span><span><i class="legend-dot weak"></i>弱线索 {{ coverageSummary.weakSignal }}</span><span><i class="legend-dot unresolved"></i>待判断 {{ coverageSummary.unresolved }}</span><span><i class="legend-dot not-found"></i>未发现 {{ coverageSummary.notFound }}</span></div></div><div class="coverage-track"><span class="track-hit" :style="{ width: `${(coverageSummary.hit / Math.max(moduleCount, 1)) * 100}%` }"></span><span class="track-weak" :style="{ width: `${(coverageSummary.weakSignal / Math.max(moduleCount, 1)) * 100}%` }"></span><span class="track-unresolved" :style="{ width: `${(coverageSummary.unresolved / Math.max(moduleCount, 1)) * 100}%` }"></span></div></div>
        <div v-if="result" class="module-groups"><div v-for="group in groupedModules" :key="group.domain" class="panel module-group"><div class="module-group-head"><span class="domain-code">{{ group.domain }}</span><div><strong>{{ group.label }}</strong><span>{{ group.modules.length }} 个模块</span></div></div><div class="module-rows"><button v-for="module in group.modules" :key="module.code" type="button" class="module-row" @click="activeTab = 'candidates'"><span class="module-status" :class="coverageStatus(module.code)"></span><span class="module-code">{{ module.code }}</span><strong>{{ module.name }}</strong><span class="module-status-label">{{ COVERAGE_LABELS[coverageStatus(module.code)] }}</span></button></div></div></div>
        <div v-else class="panel empty-state"><span class="empty-index">04</span><div><h2>等待模型结果</h2><p>未运行识别时，不能预先判断任何模块是否命中。</p></div></div>
      </section>

      <section v-else-if="activeTab === 'evaluation'" class="trace-view">
        <div class="panel trace-panel">
          <div class="section-title-row"><div><p class="eyebrow-small">PROXY EVALUATION</p><h2>独立评估与裁决边界</h2></div><span class="pill">不回写模型输出</span></div>
          <pre v-if="proxyEvaluation" class="json-viewer raw-output">{{ proxyEvaluation }}</pre>
          <p v-else class="muted">当前资料尚未登记代理评估报告。</p>
        </div>
      </section>

      <section v-else class="trace-view">
        <div v-if="result" class="trace-grid"><div class="panel trace-panel"><div class="section-title-row"><div><p class="eyebrow-small">MODEL CALLS</p><h2>模型调用轨迹</h2></div><span class="pill">{{ result.modelCalls.length }} 次</span></div><div class="trace-list"><details v-for="call in result.modelCalls" :key="`${call.attempt}-${call.purpose}`" class="trace-item" :class="callClass(call)" :open="call.attempt === 1"><summary><span class="trace-marker"></span><span class="trace-title">第 {{ call.attempt }} 次<span v-if="call.segment"> · {{ call.segment }}</span> · {{ call.purpose === 'identification' ? '识别' : call.purpose === 'output_limit_retry' ? '截断重试' : '结构修复' }}</span><span class="trace-stat">{{ call.promptTokens + call.completionTokens }} tokens</span><span class="trace-stat">{{ formatDuration(call.durationMs) }}</span><span class="trace-state">{{ call.status === 'completed' ? '完成' : '失败' }}</span></summary><div class="trace-body"><div v-if="call.error" class="alert alert-error compact-alert">{{ call.error }}</div><p v-if="call.finishReason" class="muted">finish reason: {{ call.finishReason }}</p><pre v-if="call.rawOutput" class="json-viewer trace-output">{{ prettyJson(call.rawOutput) }}</pre><p v-else class="muted">没有可展示的原始输出。</p></div></details></div></div><div class="panel trace-panel"><div class="section-title-row"><div><p class="eyebrow-small">RAW MODEL OUTPUT</p><h2>原始模型输出</h2></div><span class="pill">只读</span></div><pre class="json-viewer raw-output">{{ parsedRawOutput || '执行后显示真实模型返回。' }}</pre></div></div>
        <div v-else class="panel empty-state"><span class="empty-index">05</span><div><h2>等待一次真实模型调用</h2><p>调用轨迹、token、耗时、错误和只读原始输出都会来自后端运行结果，不在前端伪造。</p></div></div>
      </section>
    </template>
    <section v-else class="panel empty-state initial-state"><span class="empty-index">01</span><div><h2>先选择一份原始资料</h2><p>资料选择器只展示已经放入项目、完成代理解析并绑定全文 Markdown 的原件；内部证据包标识由后端维护。</p></div></section>

    <footer class="workbench-footer"><span>STKB 方案验证切片</span><span>候选知识是对象提议，不等同于正式 KnowledgeObject</span><span v-if="lastRunId" class="mono">{{ lastRunId }}</span></footer>
    </section>
  </main>
</template>
