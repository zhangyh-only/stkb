export type ProcessingMethod = 'agent_assisted' | 'capability'
export type PackageStatus = 'available' | 'unavailable'
export type CoverageStatus = 'hit' | 'weak_signal' | 'not_found' | 'unresolved'
export type ModelCallPurpose =
  | 'identification'
  | 'claim_discovery'
  | 'object_planning'
  | 'content_realization'
  | 'object_formation'
  | 'output_limit_retry'
  | 'repair'
export type ModelCallStatus = 'completed' | 'failed'
export type ProcessingStageStatus = 'completed' | 'failed'

export type SourceAnchor = {
  anchorId: string
  kind: 'page' | 'section' | 'table' | 'paragraph' | 'time_range'
  page: number | null
}

export type DocumentPackage = {
  documentPackageId: string
  workspaceId: string
  sourceFileName: string
  sourceFilePath: string
  sourceSha256: string
  fullMarkdownPath: string
  fullMarkdownSha256: string
  fullMarkdown: string
  processingMethod: ProcessingMethod
  status: PackageStatus
  anchors: SourceAnchor[]
  qualityIssues: string[]
}

export type SourceMaterial = {
  documentPackageId: string
  sourceFileName: string
  sourceFilePath: string
  sourceSha256: string
  processingMethod: ProcessingMethod
  status: PackageStatus
}

export type EntityMention = {
  mentionId: string
  text: string
  proposedType: string
  referenceRole: string
  sourceRef: string
}

export type ProposedRelation = {
  relationKind: 'entity' | 'object'
  relationType: string
  sourceRef: string
  targetRef: string
  evidence: string[]
}

export type ClaimEvidence = {
  anchorId: string
  exactQuote: string
  selector: string | null
  sourceText: string
}

export type AtomicClaim = {
  claimId: string
  claimKind: string
  statement: string
  subject: string
  attributes: Record<string, unknown>
  moduleHints: string[]
  evidence: ClaimEvidence[]
}

export type RejectedAtomicClaim = {
  claimId: string
  reasons: string[]
  rawClaim: Record<string, unknown>
}

export type CandidateObjectPlan = {
  planId: string
  title: string
  domain: string
  module: string
  objectType: string
  objectBoundary: string
  classificationBasis: string
  identityHints: Record<string, unknown>
  sourceClaimIds: string[]
}

export type RejectedObjectPlan = {
  planId: string
  reasons: string[]
  rawPlan: Record<string, unknown>
}

export type CandidateKnowledgeObject = {
  candidateId: string
  title: string
  domain: string
  module: string
  objectType: string
  objectBoundary: string
  classificationBasis: string
  identityHints: Record<string, unknown>
  plannedSourceClaimIds: string[]
  sourceClaimIds: string[]
  claimUsage: Array<{
    claimId: string
    role: 'primary' | 'supporting'
    contentPaths: string[]
    explanation: string
  }>
  contentLeafCount: number
  attributedContentLeafCount: number
  unattributedContentPaths: string[]
  qualityIssues: string[]
  content: Record<string, unknown>
  entityMentions: EntityMention[]
  evidence: string[]
  relations: ProposedRelation[]
}

export type RejectedCandidate = {
  candidateId: string
  reasons: string[]
  rawCandidate: Record<string, unknown>
}

export type CandidateNormalization = {
  candidateId: string
  field: 'domain' | 'entity_mentions' | 'relations' | 'content.expressions' | 'content.items' | 'content.resolutionElements' | 'content.attributionPruning' | 'claimUsage'
  originalValue: unknown
  normalizedValue: unknown
  reason: string
}

export type RejectedAuxiliaryItem = {
  kind: 'weak_signal' | 'unresolved_item'
  reasons: string[]
  rawItem: Record<string, unknown>
}

export type WeakSignal = {
  claimId: string | null
  module: string
  reason: string
  evidence: string[]
}

export type UnresolvedItem = {
  claimId: string | null
  description: string
  reason: string
  evidence: string[]
  module: string | null
}

export type ModelCallTrace = {
  callId: string
  stageId: string
  retryOf: string | null
  attempt: number
  purpose: ModelCallPurpose
  status: ModelCallStatus
  durationMs: number
  promptTokens: number
  completionTokens: number
  error: string | null
  systemPrompt: string | null
  userPrompt: string | null
  rawOutput: string | null
  finishReason: string | null
  segment: string | null
}

export type ProcessingStage = {
  key: string
  name: string
  status: ProcessingStageStatus
  durationMs: number
  detail: string
  actor: 'model' | 'code'
  modelCallIds: string[]
}

export type ObjectGranularityMetrics = {
  objectCount: number
  singleClaimObjectCount: number
  singleClaimObjectRate: number
  averageClaimsPerObject: number
  sourceAnchorsSplitAcrossObjects: number
}

export type StorageImpact = {
  postgresRunRecords: number
  formalKnowledgeFiles: number
  pgvectorRecords: number
  neo4jNodes: number
  neo4jRelationships: number
}

export type ModelConfigurationSnapshot = {
  temperature: number
  maxOutputTokens: number
  timeoutSeconds: number
  maxRetries: number
  maxCandidates: number
  enableThinking: boolean
  documentMaxChars: number
  maxConcurrency: number
  fingerprint: string
}

export type GoldGroupEvaluation = {
  key: string
  expectedCount: number
  minimumExpectedCount: number
  maximumExpectedCount: number | null
  predictedCount: number
  matchedCount: number
  status: 'met' | 'missed' | 'under_split_or_recall' | 'over_split' | 'contract_failed'
  predictedCandidateIds: string[]
  requiredItemCount: number | null
  predictedItemCount: number | null
  requiredContentFields: string[]
  missingContentFields: string[]
  requiredItemFields: string[]
  missingItemFields: string[]
  requiredUnresolvedEvidence: string[]
  missingUnresolvedEvidence: string[]
  requireAllEvidence: boolean
  missingExpectedEvidence: string[]
}

export type IdentificationQualityReport = {
  goldVersion: string
  goldStatus: string
  overallStatus: 'pass' | 'fail' | 'review'
  expectedObjectCount: number
  matchedExpectedCount: number
  objectRecallProxy: number
  groupsMet: number
  groupCount: number
  summaryOnlyCount: number
  evidenceBackedRate: number
  claimConsumptionRate: number
  claimAccountingRate: number
  contentAttributionRate: number
  medianContentChars: number
  groups: GoldGroupEvaluation[]
  findings: string[]
}

export type IdentificationResult = {
  runId: string
  documentPackageId: string
  status: 'completed' | 'failed'
  startedAt: string
  finishedAt: string
  durationMs: number
  provider: string
  model: string
  promptVersion: string
  schemaVersion: string
  catalogVersion: string
  catalogFingerprint: string
  rawModelOutput: string
  modelCalls: ModelCallTrace[]
  processingStages: ProcessingStage[]
  granularityMetrics: ObjectGranularityMetrics
  atomicClaims: AtomicClaim[]
  rejectedAtomicClaims: RejectedAtomicClaim[]
  objectPlans: CandidateObjectPlan[]
  rejectedObjectPlans: RejectedObjectPlan[]
  candidates: CandidateKnowledgeObject[]
  rejectedCandidates: RejectedCandidate[]
  rejectedAuxiliaryItems: RejectedAuxiliaryItem[]
  normalizations: CandidateNormalization[]
  weakSignals: WeakSignal[]
  unresolvedItems: UnresolvedItem[]
  coverageByModule: Record<string, CoverageStatus>
  callCount: number
  promptTokens: number
  completionTokens: number
  modelConfiguration: ModelConfigurationSnapshot | null
  qualityReport: IdentificationQualityReport | null
  storageImpact: StorageImpact
}

export type ResolvedBusinessEntity = {
  entityId: string
  entityType: string
  canonicalName: string
  sourceMentions: string[]
  action: 'created' | 'reused'
}

export type KnowledgeObjectEntityReference = {
  entityId: string
  referenceRole: string
  evidence: string[]
}

export type FormalKnowledgeObject = {
  knowledgeObjectId: string
  revision: number
  action: 'created' | 'updated' | 'reused' | 'review_required'
  title: string
  domain: string
  module: string
  objectType: string
  identityKey: string
  sourceLineageKeys: string[]
  contentFingerprint: string
  content: Record<string, unknown>
  entityReferences: KnowledgeObjectEntityReference[]
  evidence: string[]
  sourceCandidateIds: string[]
  sourceTraces: Array<{
    candidateId: string
    sourceClaimIds: string[]
    claimUsage: CandidateKnowledgeObject['claimUsage']
    contentLeafCount: number
    attributedContentLeafCount: number
    unattributedContentPaths: string[]
  }>
  revisionProposal: null | {
    title: string
    identityKey: string
    contentFingerprint: string
    content: Record<string, unknown>
    entityReferences: KnowledgeObjectEntityReference[]
    evidence: string[]
    sourceTraces: FormalKnowledgeObject['sourceTraces']
    changedPaths: string[]
  }
  equivalenceReason: string | null
  filePath: string
  fileSha256: string
}

export type FormalKnowledgeRelationship = {
  relationshipId: string
  relationType: string
  sourceRef: string
  sourceKind: 'knowledge_object' | 'business_entity'
  sourceRevision: number | null
  targetRef: string
  targetKind: 'knowledge_object' | 'business_entity'
  targetRevision: number | null
  direction: 'forward'
  inverseLabel: string
  scope: Record<string, unknown>
  effectivePeriod: Record<string, unknown>
  evidence: string[]
  status: 'active'
  provenance: Record<string, string>
}

export type KnowledgeFormationStage = {
  key: 'entity_resolution' | 'knowledge_merge' | 'formal_write'
  name: string
  status: 'completed' | 'pending' | 'failed'
  detail: string
}

export type KnowledgeFormationResult = {
  buildId: string
  runId: string
  documentPackageId: string
  status: 'completed' | 'review_required' | 'failed'
  entities: ResolvedBusinessEntity[]
  knowledgeObjects: FormalKnowledgeObject[]
  relationships: FormalKnowledgeRelationship[]
  stages: KnowledgeFormationStage[]
  createdCount: number
  updatedCount: number
  reusedCount: number
  reviewRequiredCount: number
  supersededCount: number
  qualityBlockedCandidateIds: string[]
  qualityBlockedCount: number
  formalKnowledgeFiles: number
}

export type KnowledgeModule = {
  domain: string
  domainName: string
  code: string
  name: string
  scope: 'core' | 'optional'
  meaning: string
  objectTypes: string[]
  coreObjects: string[]
  boundary: string
  sources: string[]
  consumers: string[]
  contentContract: {
    requiredFields: string[]
    requiredFieldsByType: Record<string, string[]>
    itemFieldsByType: Record<string, string[]>
    fieldShapesByType: Record<string, string>
    allowEmptyFields: string[]
    allowEmptyFieldsByType: Record<string, string[]>
    minimumContentChars: number
    minimumContentCharsByType: Record<string, number>
    granularity: string
    inclusion: string
    exclusion: string
    positiveExample: string
    negativeExample: string
  }
  identityContract: {
    identityFields: string[]
    identityFieldsByType: Record<string, string[]>
    sameObjectWhen: string
    differentObjectWhen: string
    mergeStrategy: string
    conflictRule: string
  }
}

export type KnowledgeDomain = {
  code: string
  name: string
  question: string
  meaning: string
  boundary: string
}

export type IdentificationCatalog = {
  version: string
  fingerprint: string
  status: 'sample_validation'
  source: string
  contentContractVersion: string
  identityContractVersion: string
  identityContractStatus: string
  scopeDefinitions: Record<'core' | 'optional', string>
  domains: KnowledgeDomain[]
  modules: KnowledgeModule[]
}

export const COVERAGE_LABELS: Record<CoverageStatus, string> = {
  hit: '已命中',
  weak_signal: '弱线索',
  not_found: '未发现',
  unresolved: '待判断',
}

export function prettyJson(value: unknown): string {
  if (typeof value === 'string') {
    try {
      return JSON.stringify(JSON.parse(value), null, 2)
    } catch {
      return value
    }
  }
  return JSON.stringify(value, null, 2)
}

export function formatDuration(durationMs: number): string {
  if (durationMs < 1_000) return `${durationMs} ms`
  return `${(durationMs / 1_000).toFixed(1)} s`
}

export function formatDate(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.valueOf())) return value
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).format(date)
}

export function coverageCount(
  coverage: Record<string, CoverageStatus> | undefined,
  status: CoverageStatus,
): number {
  return Object.values(coverage ?? {}).filter((value) => value === status).length
}
