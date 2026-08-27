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
  getDocumentPackage,
  getIdentificationCatalog,
  IdentificationApiError,
  listSourceMaterials,
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
  type SourceMaterial,
} from '../types'

type MainView = 'build' | 'knowledge' | 'rules' | 'evidence'
type BuildPhase = 'idle' | 'ready' | 'recognizing' | 'forming' | 'completed' | 'failed'

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

const selectedMaterial = computed(() =>
  sourceMaterials.value.find((item) => item.documentPackageId === selectedMaterialId.value) ?? null,
)

const selectedKnowledgeObject = computed<FormalKnowledgeObject | null>(() =>
  formation.value?.knowledgeObjects.find(
    (item) => item.knowledgeObjectId === selectedObjectId.value,
  ) ?? formation.value?.knowledgeObjects[0] ?? null,
)

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

const buildSteps = computed(() => {
  const hasSource = Boolean(documentPackage.value)
  const hasRules = Boolean(catalog.value)
  const hasIdentification = Boolean(identification.value)
  const hasFormation = Boolean(formation.value)
  return [
    { key: 'source', label: '准备资料', detail: '读取全文与来源定位', done: hasSource },
    { key: 'rules', label: '装载规则', detail: '5个销售域 / 22个模块', done: hasRules },
    { key: 'recognize', label: '识别知识', detail: '发现并组织知识内容', done: hasIdentification },
    { key: 'validate', label: '校验结果', detail: '对象边界、分类与证据', done: hasIdentification },
    { key: 'merge', label: '归一与归并', detail: '实体身份与知识身份', done: hasFormation },
    { key: 'write', label: '形成知识对象', detail: '登记并写入正式Markdown', done: hasFormation },
  ]
})

const currentStepKey = computed(() => {
  if (buildPhase.value === 'recognizing') return 'recognize'
  if (buildPhase.value === 'forming') return 'merge'
  if (!documentPackage.value) return 'source'
  if (!catalog.value) return 'rules'
  if (!identification.value) return 'recognize'
  if (!formation.value) return 'merge'
  return ''
})

const phaseLabel = computed(() => {
  const labels: Record<BuildPhase, string> = {
    idle: '选择一份资料开始',
    ready: '资料和规则已就绪',
    recognizing: '正在识别并校验销售知识',
    forming: '正在归一实体并形成知识对象',
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
    const [nextCatalog, materials] = await Promise.all([
      getIdentificationCatalog(),
      listSourceMaterials(),
    ])
    catalog.value = nextCatalog
    sourceMaterials.value = materials
  } catch (reason) {
    error.value = errorMessage(reason)
    buildPhase.value = 'failed'
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
  try {
    documentPackage.value = await getDocumentPackage(selectedMaterialId.value)
    buildPhase.value = 'ready'
    activeView.value = 'build'
  } catch (reason) {
    error.value = errorMessage(reason)
    buildPhase.value = 'failed'
  }
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
    buildPhase.value = 'completed'
    activeView.value = 'knowledge'
  } catch (reason) {
    error.value = errorMessage(reason)
    buildPhase.value = 'failed'
  }
}

function selectDomain(domainCode: string): void {
  selectedDomainCode.value = domainCode
  selectedModuleCode.value = catalog.value?.modules.find(
    (item) => item.domain === domainCode,
  )?.code ?? ''
}

function objectActionLabel(action: FormalKnowledgeObject['action']): string {
  return { created: '新增', updated: '更新', reused: '复用' }[action]
}

function callPurposeLabel(purpose: ModelCallTrace['purpose']): string {
  return {
    identification: '历史单阶段识别',
    claim_discovery: '发现原子主张',
    object_formation: '形成知识对象',
    output_limit_retry: '输出续接',
    repair: '结构修复',
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
      <div class="build-service"><i></i>服务可用</div>
    </header>

    <section class="build-shell">
      <div class="build-title-row">
        <div><p>能力验证 / 销售知识构建</p><h1>从业务资料形成可追溯的知识对象</h1></div>
        <div v-if="catalog" class="rule-brief"><span>当前规则</span><strong>{{ catalog.version }}</strong><small>5个销售域 · {{ catalog.modules.length }}个知识模块</small><button type="button" @click="activeView = 'rules'">查看规则</button></div>
      </div>

      <div v-if="error" class="build-error"><IconAlertCircle size="18" /><span>{{ error }}</span><button type="button" @click="initialize"><IconRefresh size="14" />重新检查</button></div>

      <section v-if="activeView === 'build'" class="build-view">
        <div class="build-command">
          <label><span>选择测试资料</span><select v-model="selectedMaterialId" :disabled="isBuilding" @change="selectMaterial"><option value="">请选择资料</option><option v-for="item in sourceMaterials" :key="item.documentPackageId" :value="item.documentPackageId" :disabled="item.status !== 'available'">{{ item.sourceFileName }}</option></select></label>
          <div v-if="documentPackage" class="source-inline"><IconFileDescription size="18" /><div><strong>{{ documentPackage.sourceFileName }}</strong><span>{{ documentPackage.anchors.length }}个来源位置 · {{ documentPackage.processingMethod === 'agent_assisted' ? '已完成代理解析' : '已完成解析' }}</span></div></div>
          <button class="build-primary" type="button" :disabled="!canBuild || isBuilding" @click="buildKnowledge"><IconPlayerPlay v-if="!isBuilding" size="17" fill="currentColor" /><span v-else class="build-pulse"></span>{{ isBuilding ? phaseLabel : formation ? '重新构建' : '开始构建知识' }}</button>
        </div>

        <div class="flow-board">
          <div class="flow-status"><span :class="`phase-${buildPhase}`"></span><div><strong>{{ phaseLabel }}</strong><p v-if="selectedMaterial">{{ selectedMaterial.sourceFileName }}</p></div></div>
          <ol class="flow-steps">
            <li v-for="(step, index) in buildSteps" :key="step.key" :class="{ done: step.done, current: currentStepKey === step.key }"><span><IconCheck v-if="step.done" size="14" stroke="2.4" /><template v-else>{{ index + 1 }}</template></span><div><strong>{{ step.label }}</strong><small>{{ step.detail }}</small></div></li>
          </ol>
        </div>

        <div v-if="formation" class="build-outcome">
          <div><span>本次形成</span><strong>{{ formation.knowledgeObjects.length }}</strong><em>个正式知识对象</em></div>
          <dl><div><dt>新增</dt><dd>{{ formation.createdCount }}</dd></div><div><dt>更新</dt><dd>{{ formation.updatedCount }}</dd></div><div><dt>复用</dt><dd>{{ formation.reusedCount }}</dd></div><div><dt>业务实体</dt><dd>{{ formation.entities.length }}</dd></div><div><dt>正式文件</dt><dd>{{ formation.formalKnowledgeFiles }}</dd></div></dl>
          <button type="button" @click="activeView = 'knowledge'">查看知识结果</button>
        </div>

        <div v-else class="build-guidance"><div><span>验证目标</span><h2>检查规则能否把一份资料稳定转换为正式知识对象</h2><p>一次执行会完成知识识别、分类与证据校验、业务实体归一、同身份知识归并、PostgreSQL登记和正式Markdown写入。</p></div><div class="build-boundary"><strong>本轮暂不执行</strong><span>pgvector索引</span><span>Neo4j图投影</span></div></div>
      </section>

      <section v-else-if="activeView === 'knowledge'" class="knowledge-result-view">
        <div v-if="formation" class="knowledge-result-grid">
          <aside class="knowledge-object-list"><div class="result-pane-title"><div><p>KNOWLEDGE OBJECTS</p><h2>正式知识对象</h2></div><span>{{ formation.knowledgeObjects.length }}</span></div><div class="object-list-scroll"><button v-for="item in formation.knowledgeObjects" :key="item.knowledgeObjectId" type="button" :class="{ active: selectedKnowledgeObject?.knowledgeObjectId === item.knowledgeObjectId }" @click="selectedObjectId = item.knowledgeObjectId"><span>{{ objectActionLabel(item.action) }}</span><div><strong>{{ item.title }}</strong><small>{{ item.domain }} / {{ item.module }} · {{ item.objectType }}</small></div></button></div></aside>
          <article v-if="selectedKnowledgeObject" class="formal-object-detail"><header><div><p>{{ selectedKnowledgeObject.knowledgeObjectId }} · revision {{ selectedKnowledgeObject.revision }}</p><h2>{{ selectedKnowledgeObject.title }}</h2><span>{{ selectedKnowledgeObject.domain }} / {{ selectedKnowledgeObject.module }} · {{ selectedKnowledgeObject.objectType }}</span></div><em>{{ objectActionLabel(selectedKnowledgeObject.action) }}</em></header><div class="formal-object-scroll"><section><h3>规范内容</h3><pre>{{ prettyJson(selectedKnowledgeObject.content) }}</pre></section><section><h3>业务实体引用</h3><div v-if="selectedKnowledgeObject.entityReferences.length" class="formal-refs"><div v-for="item in selectedKnowledgeObject.entityReferences" :key="`${item.entityId}-${item.referenceRole}`"><code>{{ item.entityId }}</code><strong>{{ item.referenceRole }}</strong><span>{{ item.evidence.join('、') }}</span></div></div><p v-else>该对象暂未形成业务实体引用。</p></section><section><h3>来源证据</h3><div class="formal-evidence"><code v-for="item in selectedKnowledgeObject.evidence" :key="item">{{ item }}</code></div></section><section class="formal-file"><h3>正式知识文件</h3><code>{{ selectedKnowledgeObject.filePath }}</code><span>SHA-256 {{ selectedKnowledgeObject.fileSha256 }}</span></section></div></article>
        </div>
        <div v-else class="build-empty"><IconStack2 size="30" /><h2>还没有形成正式知识对象</h2><p>返回构建流程，选择资料并完成一次知识构建。</p><button type="button" @click="activeView = 'build'">返回构建流程</button></div>
      </section>

      <section v-else-if="activeView === 'rules'" class="rules-library-view">
        <div v-if="catalog" class="rules-library-grid">
          <nav class="rules-domain-list"><div><p>SALES DOMAINS</p><h2>销售域</h2></div><button v-for="domain in catalog.domains" :key="domain.code" type="button" :class="{ active: selectedDomainCode === domain.code }" @click="selectDomain(domain.code)"><span>{{ domain.code }}</span><div><strong>{{ domain.name }}</strong><small>{{ domain.question }}</small></div></button></nav>
          <div class="rules-module-list"><div><p>{{ selectedDomain?.code }}</p><h2>{{ selectedDomain?.name }}</h2><span>{{ domainModules.length }}个模块</span></div><button v-for="module in domainModules" :key="module.code" type="button" :class="{ active: selectedModule?.code === module.code }" @click="selectedModuleCode = module.code"><code>{{ module.code }}</code><div><strong>{{ module.name }}</strong><small>{{ module.scope === 'core' ? '核心范围' : '可选范围' }}</small></div></button></div>
          <article v-if="selectedDomain && selectedModule" class="rules-detail"><header><div><p>{{ selectedModule.code }}</p><h2>{{ selectedModule.name }}</h2></div><span>{{ catalog.version }} / {{ catalog.contentContractVersion }}</span></header><div class="rules-detail-scroll"><section class="domain-definition"><b>{{ selectedDomain.question }}</b><strong>{{ selectedDomain.meaning }}</strong><p>{{ selectedDomain.boundary }}</p></section><section><h3>模块定义</h3><p>{{ selectedModule.meaning }}</p></section><section class="rule-highlight"><h3>识别与对象边界</h3><p>{{ selectedModule.contentContract.granularity }}</p></section><section class="contract-decision-grid"><div><h3>纳入条件</h3><p>{{ selectedModule.contentContract.inclusion }}</p></div><div><h3>排除条件</h3><p>{{ selectedModule.contentContract.exclusion }}</p></div></section><section><h3>内容必填字段</h3><div class="rule-type-list"><code v-for="item in selectedModule.contentContract.requiredFields" :key="item">{{ item }}</code></div><p v-if="selectedModule.contentContract.allowEmptyFields.length" class="contract-minimum">资料未明确时允许显式为空：{{ selectedModule.contentContract.allowEmptyFields.join('、') }}；不允许模型补造。</p><p class="contract-minimum">最小有效内容量：{{ selectedModule.contentContract.minimumContentChars }}字符；仅有summary不通过质量校验。</p></section><section><h3>允许形成的对象类型</h3><div class="rule-type-list"><code v-for="item in selectedModule.objectTypes" :key="item">{{ item }}</code></div></section><section class="contract-example"><div><h3>正例</h3><p>{{ selectedModule.contentContract.positiveExample }}</p></div><div><h3>反例</h3><p>{{ selectedModule.contentContract.negativeExample }}</p></div></section><section><h3>适用资料与使用方</h3><p>{{ selectedModule.sources.join('、') }}</p><p class="contract-consumers">用于：{{ selectedModule.consumers.join('、') }}</p></section></div></article>
        </div>
      </section>

      <section v-else class="run-evidence-view">
        <div v-if="identification" class="evidence-layout">
          <aside class="call-list"><div><p>MODEL EXCHANGES</p><h2>模型交互</h2></div><button v-for="(call, index) in identification.modelCalls" :key="`${call.attempt}-${call.purpose}`" type="button" :class="{ active: selectedCallIndex === index }" @click="selectedCallIndex = index"><span>{{ index + 1 }}</span><div><strong>{{ callPurposeLabel(call.purpose) }}</strong><small>{{ call.segment || '全文' }} · {{ formatDuration(call.durationMs) }}</small></div></button></aside>
          <article v-if="selectedCall" class="prompt-review"><header><div><p>{{ identification.model }} · {{ selectedCall.promptTokens + selectedCall.completionTokens }} tokens</p><h2>模型输入与响应</h2></div><span>{{ selectedCall.status === 'completed' ? '完成' : '失败' }}</span></header><div class="prompt-sections"><section><h3>系统规则</h3><pre>{{ selectedCall.systemPrompt || '旧运行未记录系统规则。' }}</pre></section><section><h3>资料输入</h3><pre>{{ selectedCall.userPrompt || '旧运行未记录资料输入。' }}</pre></section><section><h3>模型响应</h3><pre>{{ prettyJson(selectedCall.rawOutput || '') }}</pre></section></div></article>
          <aside class="run-summary"><section><span>规则版本</span><strong>{{ identification.catalogVersion }}</strong></section><section><span>提示词版本</span><strong>{{ identification.promptVersion }}</strong></section><section><span>输出合同</span><strong>{{ identification.schemaVersion }}</strong></section><section><span>中间主张</span><strong>{{ identification.atomicClaims.length }}条有效 / {{ identification.rejectedAtomicClaims.length }}条拒绝</strong></section><section v-if="identification.qualityReport"><span>Gold 对齐</span><strong>{{ identification.qualityReport.groupsMet }}/{{ identification.qualityReport.groupCount }}组 · 召回代理 {{ Math.round(identification.qualityReport.objectRecallProxy * 100) }}%</strong></section><section><span>模型参数</span><strong>temperature {{ identification.modelConfiguration?.temperature ?? '-' }}<br>max tokens {{ identification.modelConfiguration?.maxOutputTokens ?? '-' }}</strong></section><section><span>执行参数</span><strong>超时 {{ identification.modelConfiguration?.timeoutSeconds ?? '-' }}s · 重试 {{ identification.modelConfiguration?.maxRetries ?? '-' }}次<br>分段 {{ identification.modelConfiguration?.documentMaxChars ?? '-' }}字 · 并发 {{ identification.modelConfiguration?.maxConcurrency ?? '-' }}</strong></section><section><span>耗时</span><strong>{{ formatDuration(identification.durationMs) }}</strong></section><section><span>识别结果</span><strong>{{ identification.candidates.length }}项候选 / {{ identification.rejectedCandidates.length }}项拒绝</strong></section></aside>
        </div>
        <div v-else class="build-empty"><IconCode size="30" /><h2>还没有运行证据</h2><p>完成一次知识构建后，可查看模型输入、规则、资料内容和原始响应。</p><button type="button" @click="activeView = 'build'">返回构建流程</button></div>
      </section>
    </section>
  </main>
</template>
