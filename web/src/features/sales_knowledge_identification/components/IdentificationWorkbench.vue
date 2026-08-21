<script setup lang="ts">
import {
  IconActivityHeartbeat,
  IconAlertCircle,
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
  defaultDocumentPackageId,
  getDocumentPackage,
  getIdentificationCatalog,
  getIdentificationEvaluation,
  getIdentificationRun,
  IdentificationApiError,
  listIdentificationRuns,
  runIdentification,
} from '../api'
import {
  COVERAGE_LABELS,
  coverageCount,
  formatDuration,
  prettyJson,
  type CandidateKnowledge,
  type CoverageStatus,
  type DocumentPackage,
  type IdentificationResult,
  type KnowledgeModule,
  type ModelCallTrace,
  type ProcessingStage,
  type SourceAnchor,
} from '../types'

type WorkbenchTab = 'overview' | 'source' | 'candidates' | 'coverage' | 'evaluation' | 'traces'

const tabItems: { key: WorkbenchTab; label: string; icon: Component }[] = [
  { key: 'overview', label: '运行总览', icon: IconActivityHeartbeat },
  { key: 'source', label: '输入资料', icon: IconFileText },
  { key: 'candidates', label: '候选与证据', icon: IconStack2 },
  { key: 'coverage', label: '模块覆盖', icon: IconTable },
  { key: 'evaluation', label: '代理评估', icon: IconShieldCheck },
  { key: 'traces', label: '调用与阶段', icon: IconGitBranch },
]

const packageId = ref(defaultDocumentPackageId)
const documentPackage = ref<DocumentPackage | null>(null)
const result = ref<IdentificationResult | null>(null)
const selectedCandidateId = ref<string | null>(null)
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

const selectedCandidate = computed<CandidateKnowledge | null>(() => {
  const candidates = result.value?.candidates ?? []
  return candidates.find((candidate) => candidate.candidateId === selectedCandidateId.value) ?? candidates[0] ?? null
})

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
  const nextPageIndex = markdown.indexOf('\n## 第 ', contentStart)
  const contentEnd = nextPageIndex < 0 ? markdown.length : nextPageIndex
  return markdown.slice(contentStart, contentEnd).trim().slice(0, 900)
}

function candidateSummary(candidate: CandidateKnowledge): string {
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
      return `请求的资源不存在（${reason.message}）。请确认 DocumentPackage ID，或重启前端以加载最新 API 配置。`
    }
    return reason.message
  }
  if (reason instanceof Error) return reason.message
  return '请求失败，请检查 API 服务和文档包 ID。'
}

function clearError(): void {
  error.value = ''
  errorEndpoint.value = ''
}

async function probeApi(): Promise<void> {
  apiState.value = 'checking'
  try {
    const catalog = await getIdentificationCatalog()
    catalogModules.value = catalog.modules
    apiState.value = 'ready'
  } catch (reason) {
    apiState.value = 'error'
    error.value = loadErrorMessage(reason)
  }
}

async function loadPackage(): Promise<void> {
  const id = packageId.value.trim()
  if (!id) {
    error.value = '请先提供 DocumentPackage ID。'
    return
  }
  packageLoading.value = true
  clearError()
  packageNotice.value = ''
  try {
    const [nextPackage, history, evaluation, catalog] = await Promise.all([
      getDocumentPackage(id),
      listIdentificationRuns(id),
      getIdentificationEvaluation(id).catch(() => null),
      getIdentificationCatalog(),
    ])
    documentPackage.value = nextPackage
    runHistory.value = history
    result.value = history.at(-1) ?? null
    lastRunId.value = result.value?.runId ?? ''
    selectedCandidateId.value = result.value?.candidates[0]?.candidateId ?? null
    proxyEvaluation.value = evaluation?.markdown ?? ''
    catalogModules.value = catalog.modules
    apiState.value = 'ready'
    activeTab.value = 'overview'
    packageNotice.value = `已读取 ${id}`
  } catch (reason) {
    error.value = loadErrorMessage(reason)
  } finally {
    packageLoading.value = false
  }
}

async function executeRun(): Promise<void> {
  const id = packageId.value.trim()
  if (!id) {
    error.value = '请先提供 DocumentPackage ID。'
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
    result.value = nextResult
    runHistory.value = [...runHistory.value.filter((run) => run.runId !== nextResult.runId), nextResult].slice(-5)
    lastRunId.value = nextResult.runId
    selectedCandidateId.value = nextResult.candidates[0]?.candidateId ?? null
    activeTab.value = 'overview'
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
    runNotice.value = `已回读运行 ${lastRunId.value}`
  } catch (reason) {
    error.value = loadErrorMessage(reason)
  } finally {
    runLoading.value = false
  }
}

function selectCandidate(candidateId: string): void {
  selectedCandidateId.value = candidateId
  activeTab.value = 'candidates'
}

onMounted(() => {
  void probeApi()
  if (defaultDocumentPackageId) {
    void loadPackage()
  }
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
        <strong>只生成候选</strong>
        <p>本节点不写正式知识、pgvector 或 Neo4j。运行记录只进入 PostgreSQL 账本。</p>
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
          以可定位的 DocumentPackage 为输入，调用真实模型识别候选知识，并复核证据、D1-D5 覆盖和运行稳定性。
        </p>
      </div>
      <div class="header-facts" aria-label="当前验证范围">
        <div><span>输入</span><strong>DocumentPackage</strong></div>
        <div><span>调用</span><strong>真实模型</strong></div>
        <div><span>输出</span><strong>CandidateKnowledgeSet</strong></div>
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
          <span>DocumentPackage ID <small>资料证据包标识</small></span>
          <input v-model="packageId" type="text" placeholder="例如 DP-YXB-TRAINING-20260821" @keyup.enter="loadPackage" />
        </label>
        <button class="button button-quiet" type="button" :disabled="packageLoading || runLoading" @click="loadPackage">
          <IconSearch size="16" stroke="2" />
          {{ packageLoading ? '读取中…' : '读取资料' }}
        </button>
        <button class="button button-primary" type="button" :disabled="runLoading || packageLoading" @click="executeRun">
          <IconPlayerPlay v-if="!runLoading" size="16" stroke="2" />
          <span v-else class="button-loader"></span>
          {{ runLoading ? '模型识别运行中…' : '执行真实模型识别' }}
        </button>
      </div>
      <div class="control-meta">
        <span>API <code>{{ apiBaseUrl }}</code></span>
        <span>默认资料 <code>{{ defaultDocumentPackageId || '未配置' }}</code></span>
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
          <p class="muted">{{ documentPackage.documentPackageId }} · {{ documentPackage.workspaceId }}</p>
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

    <template v-if="documentPackage">
      <section v-if="activeTab === 'overview'" class="view-grid">
        <div class="main-column">
          <div v-if="result" class="metrics-grid">
            <article class="metric-card metric-accent">
              <span>候选知识</span>
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
              <p>点击“执行真实模型识别”后，后端将把全文、D1-D5 目录和证据规则组装为模型请求。没有真实运行结果前，页面不会生成候选或模拟调用数据。</p>
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

      <section v-else-if="activeTab === 'source'" class="source-view">
        <div class="source-meta-grid">
          <div class="panel source-card"><p class="eyebrow-small">SOURCE FILE</p><h2>{{ documentPackage.sourceFileName }}</h2><dl class="detail-list"><div><dt>原件 SHA-256</dt><dd class="mono hash-value">{{ documentPackage.sourceSha256 }}</dd></div><div><dt>全文路径</dt><dd class="mono">{{ documentPackage.fullMarkdownPath }}</dd></div><div><dt>全文 SHA-256</dt><dd class="mono hash-value">{{ documentPackage.fullMarkdownSha256 }}</dd></div></dl></div>
          <div class="panel source-card"><p class="eyebrow-small">QUALITY NOTES</p><h2>{{ documentPackage.qualityIssues.length ? '需要留意' : '暂无质量问题' }}</h2><ul v-if="documentPackage.qualityIssues.length" class="issue-list"><li v-for="issue in documentPackage.qualityIssues" :key="issue">{{ issue }}</li></ul><p v-else class="muted">代理解析结果未登记质量问题。</p><div class="method-stamp">{{ documentPackage.processingMethod === 'agent_assisted' ? 'AGENT ASSISTED' : 'CAPABILITY OUTPUT' }}</div></div>
        </div>
        <div class="panel source-document"><div class="section-title-row"><div><p class="eyebrow-small">FULL MARKDOWN</p><h2>可定位的全文输入</h2></div><span class="pill">只读</span></div><pre class="markdown-viewer">{{ documentPackage.fullMarkdown }}</pre></div>
        <div class="panel source-anchors"><div class="section-title-row"><div><p class="eyebrow-small">EVIDENCE ANCHORS</p><h2>来源锚点</h2></div><span class="muted">{{ documentPackage.anchors.length }} 个</span></div><div class="anchor-grid"><div v-for="anchor in documentPackage.anchors" :key="anchor.anchorId" class="anchor-item"><code>{{ anchor.anchorId }}</code><span>{{ anchor.kind }}<template v-if="anchor.page"> · 第 {{ anchor.page }} 页</template></span></div></div></div>
      </section>

      <section v-else-if="activeTab === 'candidates'" class="candidate-view">
        <template v-if="result">
          <div class="candidate-layout">
            <div class="candidate-list panel">
              <div class="section-title-row"><div><p class="eyebrow-small">VALIDATED CANDIDATES</p><h2>候选知识</h2></div><span class="pill pill-green">{{ result.candidates.length }} 通过</span></div>
              <button v-for="candidate in result.candidates" :key="candidate.candidateId" type="button" class="candidate-list-item" :class="{ selected: selectedCandidate?.candidateId === candidate.candidateId }" @click="selectCandidate(candidate.candidateId)"><span class="candidate-id">{{ candidate.candidateId }}</span><span class="candidate-text">{{ candidateSummary(candidate) }}</span><span class="candidate-module">{{ candidate.module }} · {{ candidate.objectType }}</span></button>
              <div v-if="!result.candidates.length" class="empty-inline">本次运行没有通过校验的候选。</div>
            </div>
            <div class="candidate-detail">
              <div v-if="selectedCandidate" class="panel detail-card">
                <div class="detail-head"><div><p class="eyebrow-small">{{ selectedCandidate.candidateId }} / {{ selectedCandidate.domain }}</p><h2>{{ selectedCandidate.module }} <span>· {{ selectedCandidate.objectType }}</span></h2></div><span class="pill pill-green">证据有效</span></div>
                <div class="content-block"><h3>类型化内容</h3><pre class="json-viewer">{{ prettyJson(selectedCandidate.content) }}</pre></div>
                <div class="content-block"><h3>来源证据与原文片段</h3><div class="evidence-list"><div v-for="reference in selectedCandidate.evidence" :key="reference" class="evidence-item"><code>{{ evidenceLabel(reference) }}</code><span v-if="anchorFor(reference)" class="evidence-ok">可解析</span><span v-else class="evidence-missing">未找到锚点</span><p v-if="evidenceExcerpt(reference)" class="evidence-excerpt">{{ evidenceExcerpt(reference) }}</p></div></div></div>
                <div v-if="selectedCandidate.entityMentions.length" class="content-block"><h3>实体提及与引用角色</h3><div class="mention-list"><div v-for="mention in selectedCandidate.entityMentions" :key="mention.mentionId" class="mention-item"><strong>{{ mention.text }}</strong><span>{{ mention.proposedType }} · {{ mention.referenceRole }}</span><code>{{ mention.sourceRef }}</code></div></div></div>
                <div v-if="selectedCandidate.relations.length" class="content-block"><h3>关系建议</h3><div class="relation-list"><div v-for="(relation, index) in selectedCandidate.relations" :key="`${relation.relationType}-${index}`"><span class="relation-kind">{{ relation.relationKind }}</span><strong>{{ relation.sourceRef }}</strong><span>→</span><strong>{{ relation.targetRef }}</strong><span class="muted">{{ relation.relationType }}</span></div></div></div>
              </div>
              <div v-else class="panel empty-state"><span class="empty-index">03</span><div><h2>暂无候选详情</h2><p>模型结果中的有效候选会在这里和原文证据并排展示。</p></div></div>
            </div>
          </div>
          <div class="secondary-result-grid"><div class="panel secondary-card"><div class="section-title-row"><div><p class="eyebrow-small">REJECTED</p><h2>拒绝项</h2></div><span class="pill pill-red">{{ result.rejectedCandidates.length }}</span></div><div v-if="result.rejectedCandidates.length" class="rejected-list"><details v-for="item in result.rejectedCandidates" :key="item.candidateId"><summary><strong>{{ item.candidateId }}</strong><span>{{ item.reasons.length }} 个原因</span></summary><ul class="issue-list"><li v-for="reason in item.reasons" :key="reason">{{ reason }}</li></ul><pre class="json-viewer compact">{{ prettyJson(item.rawCandidate) }}</pre></details></div><p v-else class="muted">没有被程序拒绝的候选。</p></div><div class="panel secondary-card"><div class="section-title-row"><div><p class="eyebrow-small">WEAK / UNRESOLVED</p><h2>弱线索与未决项</h2></div><span class="pill">{{ result.weakSignals.length + result.unresolvedItems.length }}</span></div><div v-if="result.weakSignals.length || result.unresolvedItems.length" class="weak-list"><div v-for="(item, index) in result.weakSignals" :key="`weak-${index}`" class="weak-item"><span class="pill pill-amber">弱线索 · {{ item.module }}</span><p>{{ item.reason }}</p><div class="evidence-chips"><code v-for="reference in item.evidence" :key="reference">{{ reference }}</code></div></div><div v-for="(item, index) in result.unresolvedItems" :key="`unresolved-${index}`" class="weak-item"><span class="pill pill-purple">未决{{ item.module ? ` · ${item.module}` : '' }}</span><p>{{ item.description }}：{{ item.reason }}</p></div></div><p v-else class="muted">本次没有弱线索或未决项。</p></div></div>
          <div v-if="result.normalizations.length" class="panel secondary-card"><div class="section-title-row"><div><p class="eyebrow-small">DETERMINISTIC NORMALIZATION</p><h2>公开规范化记录</h2></div><span class="pill pill-amber">{{ result.normalizations.length }}</span></div><div class="weak-list"><div v-for="item in result.normalizations" :key="`${item.candidateId}-${item.field}`" class="weak-item"><strong>{{ item.candidateId }}</strong><p>{{ item.field }}：{{ item.originalValue }} → {{ item.normalizedValue }}</p><span class="muted">{{ item.reason }}</span></div></div></div>
        </template>
        <div v-else class="panel empty-state"><span class="empty-index">03</span><div><h2>运行后查看候选与证据</h2><p>当前只加载了 DocumentPackage。执行一次真实模型识别后，这里会展示通过项、拒绝项、弱线索和未决项。</p></div></div>
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
          <p v-else class="muted">当前 DocumentPackage 尚未登记代理评估报告。</p>
        </div>
      </section>

      <section v-else class="trace-view">
        <div v-if="result" class="trace-grid"><div class="panel trace-panel"><div class="section-title-row"><div><p class="eyebrow-small">MODEL CALLS</p><h2>模型调用轨迹</h2></div><span class="pill">{{ result.modelCalls.length }} 次</span></div><div class="trace-list"><details v-for="call in result.modelCalls" :key="`${call.attempt}-${call.purpose}`" class="trace-item" :class="callClass(call)" :open="call.attempt === 1"><summary><span class="trace-marker"></span><span class="trace-title">第 {{ call.attempt }} 次<span v-if="call.segment"> · {{ call.segment }}</span> · {{ call.purpose === 'identification' ? '识别' : call.purpose === 'output_limit_retry' ? '截断重试' : '结构修复' }}</span><span class="trace-stat">{{ call.promptTokens + call.completionTokens }} tokens</span><span class="trace-stat">{{ formatDuration(call.durationMs) }}</span><span class="trace-state">{{ call.status === 'completed' ? '完成' : '失败' }}</span></summary><div class="trace-body"><div v-if="call.error" class="alert alert-error compact-alert">{{ call.error }}</div><p v-if="call.finishReason" class="muted">finish reason: {{ call.finishReason }}</p><pre v-if="call.rawOutput" class="json-viewer trace-output">{{ prettyJson(call.rawOutput) }}</pre><p v-else class="muted">没有可展示的原始输出。</p></div></details></div></div><div class="panel trace-panel"><div class="section-title-row"><div><p class="eyebrow-small">RAW MODEL OUTPUT</p><h2>原始模型输出</h2></div><span class="pill">只读</span></div><pre class="json-viewer raw-output">{{ parsedRawOutput || '执行后显示真实模型返回。' }}</pre></div></div>
        <div v-else class="panel empty-state"><span class="empty-index">05</span><div><h2>等待一次真实模型调用</h2><p>调用轨迹、token、耗时、错误和只读原始输出都会来自后端运行结果，不在前端伪造。</p></div></div>
      </section>
    </template>
    <section v-else class="panel empty-state initial-state"><span class="empty-index">01</span><div><h2>先读取一份 DocumentPackage</h2><p>默认 ID：{{ defaultDocumentPackageId || '尚未配置' }}。输入文档包 ID 后读取全文和来源锚点，再执行识别。</p></div></section>

    <footer class="workbench-footer"><span>STKB 方案验证切片</span><span>候选知识不等同于正式知识</span><span v-if="lastRunId" class="mono">{{ lastRunId }}</span></footer>
    </section>
  </main>
</template>
