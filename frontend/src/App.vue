<script setup lang="ts">
/**
 * App.vue — Pure Edge Mode
 * =========================
 * PDF → PDF.js [browser] → ONNX Worker (SciBERT INT8) → Evidence Units JSON → /api/chat [backend proxy]
 * 
 * No server-side upload. No fallback. All inference runs in the browser via WebAssembly.
 * The backend is a pure LLM proxy that receives structured Evidence Units JSON.
 */
import { ref, nextTick, watch, onMounted, onBeforeUnmount, computed } from 'vue'
import { marked } from 'marked'

// ─── Types ────────────────────────────────────────────────────────────────────

interface ThresholdTrace {
  ngram_length: {
    formula: string
    phrase_word_number: number
    threshold: number
    passed: boolean
  }
  bow_support: {
    formula: string
    term_confidence: number
    match_quality: number
    bow_support_score: number
    threshold: number
    passed: boolean
  }
  tfidf_support: {
    formula: string
    matched_tfidf: number
    max_tfidf: number
    tfidf_support_score: number
    threshold: number
    passed: boolean
  }
  passing_rule: string
  passed: boolean
  passed_features: string[]
  failed_features: string[]
}

interface ConceptUnit {
  section: string
  phrase: string
  words?: string[]
  role: string
  evidence_sentence: string
  importance: number
  sentence_index: number
  s_boundary: number
  s_selector: number
  s_bow: number
  s_candidate: number
  s_coverage: number
  s_rerank: number
  i_concept: number
  boundary_score: number
  evidence_score: number
  role_score: number
  sentence_importance_score: number
  threshold_trace: ThresholdTrace
}

interface EvidencePayload {
  title: string
  total_sentences: number
  sections_found: string[]
  units: ConceptUnit[]
  section_notes: Record<string, string[]>
  chunk_text: string[]
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface ContextChunk {
  text: string
  phrase: string | null
  section: string | null
  section_label: string
  role: string | null
  score: number
  importance: number | null
  boundary_score: number | null
  evidence_score: number | null
  matched_keywords: string[]
  matched_bow_terms?: BowTerm[]
  source: 'edge'
}

interface BowTerm {
  term: string
  canonical_term: string
  matched_alias: string
  category: string
  domain?: string
  wiki_url: string
  wikidata_id: string
  document_frequency: string | number
  total_frequency: string | number
  confidence: number
  match_score: number
}

interface SemanticKgEdge {
  source: string
  target: string
  relation: string
  evidence: string
  section?: string
  role?: string
  source_type?: string
}

interface KgConceptDetail {
  label: string
  category: string
  df: string | number
  tf: string | number
  confidence: number
  aliases: string
  wikidata: string
  wikidataUrl: string
}

interface DynamicKgNode {
  id: string
  label: string
  group: 'domain' | 'term'
  color: string
  term?: BowTerm
  x: number
  y: number
  vx: number
  vy: number
  fixed?: boolean
}

interface DynamicKgEdge {
  source: DynamicKgNode
  target: DynamicKgNode
  label: string
  type: 'domain' | 'related' | 'semantic'
  evidence?: string
}

type RightView = 'chat' | 'graph'

// ─── Utilities ────────────────────────────────────────────────────────────────

const renderMessage = (text: string): string => {
  return marked.parse(text) as string
}

function highlightText(text: string, keywords: string[]): string {
  let escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  if (!keywords.length) return escaped
  const sorted = [...keywords].sort((a, b) => b.length - a.length)
  for (const kw of sorted) {
    const regex = new RegExp(`(${kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
    escaped = escaped.replace(regex, '<mark class="bow-hit">$1</mark>')
  }
  return escaped
}

// ─── Constants ────────────────────────────────────────────────────────────────

const SECTION_DISPLAY: Record<string, string> = {
  intro: 'Introduction',
  related_work: 'Related Work',
  method: 'Methodology',
  experiment: 'Experiments',
  conclusion: 'Conclusion',
}

const ROLE_BADGES: Record<string, { label: string; color: string }> = {
  background:   { label: 'Background',   color: '#6366f1' },
  problem:      { label: 'Problem',      color: '#ef4444' },
  motivation:   { label: 'Motivation',   color: '#f59e0b' },
  objective:    { label: 'Objective',    color: '#0ea5e9' },
  prior_work:   { label: 'Prior Work',   color: '#8b5cf6' },
  limitation:   { label: 'Limitation',   color: '#f97316' },
  gap:          { label: 'Gap',          color: '#ec4899' },
  core_method:  { label: 'Method',       color: '#10b981' },
  component:    { label: 'Component',    color: '#14b8a6' },
  mechanism:    { label: 'Mechanism',    color: '#06b6d4' },
  dataset:      { label: 'Dataset',      color: '#3b82f6' },
  metric:       { label: 'Metric',       color: '#a855f7' },
  result:       { label: 'Result',       color: '#22c55e' },
  contribution: { label: 'Contribution', color: '#4f46e5' },
  finding:      { label: 'Finding',      color: '#84cc16' },
}

function getRoleColor(role: string): string {
  return ROLE_BADGES[role]?.color ?? '#64748b'
}

const GLOBAL_KG_CATEGORIES = [
  {
    id: 'machine-learning',
    label: 'Machine Learning',
    x: 130,
    y: 110,
    color: '#4f46e5',
    concepts: [
      { label: 'training set', x: 80, y: 245, df: 102, tf: 878, confidence: 0.72, wikidata: 'Q3997298', aliases: 'training data; training dataset; training set; training-set' },
      { label: 'gradient descent', x: 185, y: 260, df: 83, tf: 215, confidence: 0.82, wikidata: 'Q1199743', aliases: 'gradient descent' },
      { label: 'reinforcement learning', x: 250, y: 160, df: 42, tf: 287, confidence: 0.92, wikidata: 'Q830687', aliases: 'reinforcement learning' },
    ],
  },
  {
    id: 'neural-network',
    label: 'Artificial Neural Network',
    x: 445,
    y: 95,
    color: '#0ea5e9',
    concepts: [
      { label: 'attention', x: 355, y: 225, df: 85, tf: 875, confidence: 0.72, wikidata: 'Q103701642', aliases: 'attention; attention mechanism' },
      { label: 'transformer', x: 470, y: 260, df: 53, tf: 819, confidence: 0.92, wikidata: 'Q85810444', aliases: 'transformer; transformer architecture; transformer model; transformer models; transformers' },
      { label: 'convolutional neural network', x: 590, y: 185, df: 41, tf: 258, confidence: 0.92, wikidata: 'Q17084460', aliases: 'convolutional neural network; convolutional neural networks; cnn' },
    ],
  },
  {
    id: 'nlp-llm',
    label: 'NLP & LLM',
    x: 735,
    y: 120,
    color: '#10b981',
    concepts: [
      { label: 'embedding', x: 675, y: 255, df: 82, tf: 1192, confidence: 0.72, wikidata: 'Q133284072', aliases: 'embedding; embeddings' },
      { label: 'BERT', x: 785, y: 270, df: 48, tf: 796, confidence: 0.92, wikidata: 'Q61713173', aliases: 'BERT; bidirectional encoder representations from transformers' },
      { label: 'question answering', x: 825, y: 175, df: 47, tf: 306, confidence: 0.92, wikidata: 'Q1074173', aliases: 'question answering; question-answering' },
    ],
  },
]

const KG_DOMAIN_COLORS: Record<string, string> = {
  'NLP & LLM': '#10b981',
  'Deep Learning': '#0ea5e9',
  'Information Retrieval': '#f59e0b',
  'Optimization': '#ef4444',
  'Evaluation': '#8b5cf6',
  'Computer Vision': '#3b82f6',
  'Graph Learning': '#14b8a6',
  'Reinforcement Learning': '#84cc16',
  'Data & Training': '#64748b',
  Other: '#64748b',
}

// ─── State ────────────────────────────────────────────────────────────────────

// Worker & inference
let worker: Worker | null = null
const workerReady = ref(false)
const workerError = ref('')

// Extraction
const isExtracting = ref(false)
const extractionProgress = ref(0)
const extractionStage = ref('')
const evidencePayload = ref<EvidencePayload | null>(null)
const paperTitle = ref('')
const totalSentences = ref(0)
const sectionsFound = ref<string[]>([])

// Left panel tab
const activeSection = ref<string | null>(null)

// Chat
const messages = ref<ChatMessage[]>([
  {
    role: 'assistant',
    content: 'Hello! Upload a scientific paper (PDF) and I will extract its structure and concept units locally in your browser — then you can ask me anything about it.',
  },
])
const currentInput = ref('')
const isGenerating = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
const chatContainer = ref<HTMLDivElement | null>(null)

// Context Visualization
const contextChunks = ref<ContextChunk[]>([])
const queryKeywords = ref<string[]>([])
const termExplanations = ref<BowTerm[]>([])
const kgSemanticEdges = ref<SemanticKgEdge[]>([])
const isContextOpen = ref(true)
const lastQuery = ref('')
const activeRightView = ref<RightView>('chat')
const kgCanvas = ref<HTMLCanvasElement | null>(null)
const kgTooltip = ref<HTMLElement | null>(null)
const selectedKgConcept = ref<KgConceptDetail>({
  label: 'attention',
  category: 'Artificial Neural Network',
  df: 85,
  tf: 875,
  confidence: 0.72,
  aliases: 'attention; attention mechanism',
  wikidata: 'Q103701642',
  wikidataUrl: 'https://www.wikidata.org/wiki/Q103701642',
})

// ─── Computed ─────────────────────────────────────────────────────────────────

const filteredUnits = computed(() => {
  if (!evidencePayload.value) return []
  const units = evidencePayload.value.units
  if (!activeSection.value) return units
  return units.filter(u => u.section === activeSection.value)
})

const sectionTabs = computed(() => {
  if (!evidencePayload.value) return []
  return evidencePayload.value.sections_found.map(s => ({
    id: s,
    label: SECTION_DISPLAY[s] ?? s,
  }))
})

const hasPaper = computed(() => !!evidencePayload.value)

const hasBagOfWordGraph = computed(() => termExplanations.value.length > 0)
const graphConceptCount = computed(() => termExplanations.value.length)
const graphCategoryCount = computed(() => {
  return new Set(termExplanations.value.map(t => getBowDomain(t))).size
})

function slugify(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'node'
}

function normalizeKgKey(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9+\-]+/g, ' ').replace(/\s+/g, ' ').trim()
}

function getBowDomain(term: BowTerm): string {
  if (term.domain) return term.domain
  const text = `${term.canonical_term} ${term.term} ${term.category}`.toLowerCase()
  if (/natural language|language model|large language|bert|transformer|text|token|translation|question answering|embedding/.test(text)) return 'NLP & LLM'
  if (/deep learning|neural network|artificial neural|attention|backpropagation|convolutional|activation|layer/.test(text)) return 'Deep Learning'
  if (/information retrieval|retrieval|rag|search|ranking|passage|query|bm25/.test(text)) return 'Information Retrieval'
  if (/optimization|gradient|loss|optimizer|regularization|descent|learning rate/.test(text)) return 'Optimization'
  if (/evaluation|metric|benchmark|accuracy|bleu|rouge|f1/.test(text)) return 'Evaluation'
  if (/computer vision|image|vision|object detection/.test(text)) return 'Computer Vision'
  if (/graph|node|edge|knowledge graph/.test(text)) return 'Graph Learning'
  if (/reinforcement|policy|reward|q-learning|actor/.test(text)) return 'Reinforcement Learning'
  if (/training data|training set|dataset|corpus|pre-training/.test(text)) return 'Data & Training'
  return term.category || 'Other'
}

function getWikidataUrl(wikidataId: string, wikiUrl?: string): string {
  if (wikiUrl) return wikiUrl
  if (!wikidataId || wikidataId === '—') return ''
  return `https://www.wikidata.org/wiki/${encodeURIComponent(wikidataId)}`
}

function selectBowTerm(term: BowTerm) {
  const wikidataId = term.wikidata_id || '—'
  selectedKgConcept.value = {
    label: term.canonical_term || term.term,
    category: term.domain || term.category || 'Matched terminology',
    df: term.document_frequency || '—',
    tf: term.total_frequency || '—',
    confidence: term.confidence || term.match_score || 0,
    aliases: term.matched_alias,
    wikidata: wikidataId,
    wikidataUrl: getWikidataUrl(wikidataId, term.wiki_url),
  }
  drawKgCanvas()
}

let kgNodes: DynamicKgNode[] = []
let kgEdges: DynamicKgEdge[] = []
let kgWidth = 0
let kgHeight = 0
let kgScale = 1
let kgOffsetX = 0
let kgOffsetY = 0
let kgDraggingNode: DynamicKgNode | null = null
let kgPanning = false
let kgLastPointer: { x: number; y: number } | null = null
let kgAnimationFrame = 0
let kgAlpha = 0

const KG_MIN_ALPHA = 0.018
const KG_MAX_SPEED = 5.5

function kgNodeRadius(node: DynamicKgNode): number {
  return node.group === 'domain' ? 34 : 18
}

function buildDynamicKg() {
  const grouped = new Map<string, BowTerm[]>()
  for (const term of termExplanations.value) {
    const domain = getBowDomain(term)
    const list = grouped.get(domain) ?? []
    list.push(term)
    grouped.set(domain, list)
  }

  kgNodes = []
  kgEdges = []
  const nodeByTerm = new Map<string, DynamicKgNode>()

  const domains = Array.from(grouped.keys())
  const allTerms = domains.flatMap(domain => grouped.get(domain) ?? [])
  const termCount = Math.max(allTerms.length, 1)
  domains.forEach((domain, domainIdx) => {
    const color = KG_DOMAIN_COLORS[domain] ?? KG_DOMAIN_COLORS.Other
    const terms = grouped.get(domain) ?? []
    terms.forEach((term, termIdx) => {
      const globalIdx = allTerms.findIndex(item => (item.canonical_term || item.term) === (term.canonical_term || term.term))
      const angle = (Math.PI * 2 * Math.max(globalIdx, 0)) / termCount - Math.PI / 2
      const domainOffset = (domainIdx - (domains.length - 1) / 2) * 26
      const ring = 150 + (termIdx % 3) * 42
      const termNode: DynamicKgNode = {
        id: `term:${slugify(domain)}:${slugify(term.canonical_term || term.term)}`,
        label: term.canonical_term || term.term,
        group: 'term',
        color,
        term,
        x: Math.cos(angle) * ring + domainOffset,
        y: Math.sin(angle) * ring + domainOffset * 0.35,
        vx: 0,
        vy: 0,
      }
      kgNodes.push(termNode)
      nodeByTerm.set(normalizeKgKey(term.canonical_term || term.term), termNode)
    })
  })

  for (const edge of kgSemanticEdges.value) {
    const source = nodeByTerm.get(normalizeKgKey(edge.source))
    const target = nodeByTerm.get(normalizeKgKey(edge.target))
    if (!source || !target || source.id === target.id) continue
    kgEdges.push({
      source,
      target,
      label: edge.relation,
      type: 'semantic',
      evidence: edge.evidence,
    })
  }

  for (const domain of domains) {
    const termNodes = kgNodes.filter(node => node.term && getBowDomain(node.term) === domain)
    for (let idx = 0; idx < termNodes.length - 1; idx++) {
      const source = termNodes[idx]
      const target = termNodes[idx + 1]
      const hasSemanticEdge = kgEdges.some(edge =>
        edge.type === 'semantic' &&
        ((edge.source.id === source.id && edge.target.id === target.id) ||
          (edge.source.id === target.id && edge.target.id === source.id))
      )
      if (hasSemanticEdge) continue
      kgEdges.push({ source: termNodes[idx], target: termNodes[idx + 1], label: 'related', type: 'related' })
    }
  }
}

function resizeKgCanvas() {
  const canvas = kgCanvas.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const ratio = window.devicePixelRatio || 1
  kgWidth = rect.width
  kgHeight = rect.height
  canvas.width = Math.floor(kgWidth * ratio)
  canvas.height = Math.floor(kgHeight * ratio)
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0)
  if (!kgOffsetX && !kgOffsetY) {
    kgOffsetX = kgWidth / 2
    kgOffsetY = kgHeight / 2
  }
  drawKgCanvas()
}

function kgToScreen(node: DynamicKgNode) {
  return { x: node.x * kgScale + kgOffsetX, y: node.y * kgScale + kgOffsetY }
}

function kgToWorld(x: number, y: number) {
  return { x: (x - kgOffsetX) / kgScale, y: (y - kgOffsetY) / kgScale }
}

function drawKgLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, maxWidth: number) {
  const clean = text.length > 26 ? `${text.slice(0, 25)}...` : text
  ctx.fillText(clean, x, y, maxWidth)
}

function kgEdgePairKey(edge: DynamicKgEdge): string {
  return [edge.source.id, edge.target.id].sort().join('::')
}

function drawKgCanvas() {
  const canvas = kgCanvas.value
  const ctx = canvas?.getContext('2d')
  if (!canvas || !ctx) return
  ctx.clearRect(0, 0, kgWidth, kgHeight)

  const semanticParallelEdges = new Map<string, DynamicKgEdge[]>()
  for (const edge of kgEdges) {
    if (edge.type !== 'semantic') continue
    const key = kgEdgePairKey(edge)
    const group = semanticParallelEdges.get(key) ?? []
    group.push(edge)
    semanticParallelEdges.set(key, group)
  }

  kgEdges.forEach(edge => {
    const source = kgToScreen(edge.source)
    const target = kgToScreen(edge.target)
    const parallelGroup = edge.type === 'semantic'
      ? semanticParallelEdges.get(kgEdgePairKey(edge)) ?? []
      : []
    const parallelIndex = parallelGroup.indexOf(edge)
    const parallelCount = parallelGroup.length
    const dx = target.x - source.x
    const dy = target.y - source.y
    const dist = Math.max(1, Math.hypot(dx, dy))
    const normalX = -dy / dist
    const normalY = dx / dist
    const curveOffset = edge.type === 'semantic' && parallelCount > 1
      ? (parallelIndex - (parallelCount - 1) / 2) * 28
      : 0
    const midX = (source.x + target.x) / 2
    const midY = (source.y + target.y) / 2
    const controlX = midX + normalX * curveOffset
    const controlY = midY + normalY * curveOffset

    ctx.beginPath()
    ctx.moveTo(source.x, source.y)
    if (edge.type === 'semantic' && parallelCount > 1) {
      ctx.quadraticCurveTo(controlX, controlY, target.x, target.y)
    } else {
      ctx.lineTo(target.x, target.y)
    }
    ctx.strokeStyle = edge.type === 'domain'
      ? 'rgba(148, 163, 184, 0.42)'
      : edge.type === 'semantic'
        ? `${edge.source.color}d9`
        : `${edge.source.color}70`
    ctx.lineWidth = edge.type === 'semantic' ? 2.2 : edge.type === 'domain' ? 1.4 : 1.1
    if (edge.type === 'related') ctx.setLineDash([5, 5])
    ctx.stroke()
    ctx.setLineDash([])
    if (edge.type === 'semantic' && kgScale > 0.58) {
      ctx.font = '700 10px Arial, sans-serif'
      ctx.fillStyle = 'rgba(71, 85, 105, 0.86)'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'bottom'
      const labelX = controlX + normalX * (parallelCount > 1 ? 6 : 0)
      const labelY = controlY + normalY * (parallelCount > 1 ? 6 : 0)
      ctx.fillText(edge.label.replace(/_/g, ' '), labelX, labelY - 4, 102)
    }
  })

  kgNodes.forEach(node => {
    const point = kgToScreen(node)
    const radius = kgNodeRadius(node)
    const isSelected = selectedKgConcept.value.label === node.label

    ctx.beginPath()
    ctx.arc(point.x, point.y, isSelected ? radius + 4 : radius, 0, Math.PI * 2)
    ctx.fillStyle = node.group === 'domain' ? node.color : `${node.color}33`
    ctx.strokeStyle = node.group === 'domain' ? '#ffffff' : node.color
    ctx.lineWidth = isSelected ? 3 : 1.8
    ctx.fill()
    ctx.stroke()

    ctx.font = node.group === 'domain' ? '700 12px Arial, sans-serif' : '700 11px Arial, sans-serif'
    ctx.fillStyle = '#1f2937'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    drawKgLabel(ctx, node.label, point.x, point.y + radius + 6, node.group === 'domain' ? 110 : 126)
  })
}

function tickKgCanvas() {
  if (kgAlpha < KG_MIN_ALPHA) {
    kgAlpha = 0
    drawKgCanvas()
    return
  }

  for (let step = 0; step < 2; step++) {
    kgEdges.forEach(edge => {
      const a = edge.source
      const b = edge.target
      const dx = b.x - a.x
      const dy = b.y - a.y
      const dist = Math.max(1, Math.hypot(dx, dy))
      const target = edge.type === 'domain' ? 150 : edge.type === 'semantic' ? 128 : 116
      const force = (dist - target) * 0.0025 * kgAlpha
      const fx = (dx / dist) * force
      const fy = (dy / dist) * force
      if (!a.fixed) {
        a.vx += fx
        a.vy += fy
      }
      if (!b.fixed) {
        b.vx -= fx
        b.vy -= fy
      }
    })

    for (let i = 0; i < kgNodes.length; i++) {
      for (let j = i + 1; j < kgNodes.length; j++) {
        const a = kgNodes[i]
        const b = kgNodes[j]
        const dx = b.x - a.x
        const dy = b.y - a.y
        const minDist = kgNodeRadius(a) + kgNodeRadius(b) + 44
        const distSq = Math.max(120, dx * dx + dy * dy)
        const dist = Math.sqrt(distSq)
        const force = Math.min(1.25, (minDist * minDist * 0.5) / distSq) * kgAlpha
        const fx = (dx / dist) * force
        const fy = (dy / dist) * force
        if (!a.fixed) {
          a.vx -= fx
          a.vy -= fy
        }
        if (!b.fixed) {
          b.vx += fx
          b.vy += fy
        }
      }
    }

    kgNodes.forEach(node => {
      if (node.fixed) return
      node.vx += -node.x * 0.0018 * kgAlpha
      node.vy += -node.y * 0.0018 * kgAlpha
      const speed = Math.hypot(node.vx, node.vy)
      if (speed > KG_MAX_SPEED) {
        node.vx = (node.vx / speed) * KG_MAX_SPEED
        node.vy = (node.vy / speed) * KG_MAX_SPEED
      }
      node.vx *= 0.76
      node.vy *= 0.76
      node.x += node.vx
      node.y += node.vy
      node.x = Math.max(-360, Math.min(360, node.x))
      node.y = Math.max(-230, Math.min(230, node.y))
    })
  }
  kgAlpha *= 0.94
  drawKgCanvas()
  kgAnimationFrame = requestAnimationFrame(tickKgCanvas)
}

function nearestKgNode(clientX: number, clientY: number): DynamicKgNode | null {
  const canvas = kgCanvas.value
  if (!canvas) return null
  const rect = canvas.getBoundingClientRect()
  const x = clientX - rect.left
  const y = clientY - rect.top
  let best: DynamicKgNode | null = null
  let bestDist = Infinity
  for (const node of kgNodes) {
    const point = kgToScreen(node)
    const d = Math.hypot(point.x - x, point.y - y)
    if (d < kgNodeRadius(node) + 10 && d < bestDist) {
      best = node
      bestDist = d
    }
  }
  return best
}

function setupKgCanvas() {
  if (!termExplanations.value.length || !kgCanvas.value) return
  cancelAnimationFrame(kgAnimationFrame)
  buildDynamicKg()
  kgScale = 1
  kgOffsetX = 0
  kgOffsetY = 0
  kgAlpha = 1
  resizeKgCanvas()
  tickKgCanvas()
}

function restartKgSimulation(alpha = 0.35) {
  kgAlpha = Math.max(kgAlpha, alpha)
  cancelAnimationFrame(kgAnimationFrame)
  tickKgCanvas()
}

function handleKgPointerDown(event: PointerEvent) {
  const canvas = kgCanvas.value
  if (!canvas) return
  const node = nearestKgNode(event.clientX, event.clientY)
  const rect = canvas.getBoundingClientRect()
  kgLastPointer = { x: event.clientX - rect.left, y: event.clientY - rect.top }
  if (node) {
    kgDraggingNode = node
    node.fixed = true
    if (node.term) selectBowTerm(node.term)
  } else {
    kgPanning = true
  }
  canvas.setPointerCapture(event.pointerId)
}

function handleKgPointerMove(event: PointerEvent) {
  const canvas = kgCanvas.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top
  const node = nearestKgNode(event.clientX, event.clientY)

  if (kgDraggingNode) {
    const world = kgToWorld(x, y)
    kgDraggingNode.x = world.x
    kgDraggingNode.y = world.y
    kgDraggingNode.vx = 0
    kgDraggingNode.vy = 0
    restartKgSimulation(0.18)
  } else if (kgPanning && kgLastPointer) {
    kgOffsetX += x - kgLastPointer.x
    kgOffsetY += y - kgLastPointer.y
    drawKgCanvas()
  }
  kgLastPointer = { x, y }

  if (kgTooltip.value) {
    if (node && !kgDraggingNode && !kgPanning) {
      kgTooltip.value.style.display = 'block'
      kgTooltip.value.style.left = `${x}px`
      kgTooltip.value.style.top = `${y}px`
      kgTooltip.value.textContent = node.term
        ? `${node.label} · ${getBowDomain(node.term)}`
        : node.label
    } else {
      kgTooltip.value.style.display = 'none'
    }
  }
}

function handleKgPointerUp(event: PointerEvent) {
  const canvas = kgCanvas.value
  if (kgDraggingNode) kgDraggingNode.fixed = false
  kgDraggingNode = null
  kgPanning = false
  kgLastPointer = null
  if (kgNodes.length) restartKgSimulation(0.28)
  if (canvas?.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId)
  }
  if (kgTooltip.value) kgTooltip.value.style.display = 'none'
}

function handleKgWheel(event: WheelEvent) {
  event.preventDefault()
  const canvas = kgCanvas.value
  if (!canvas) return
  const rect = canvas.getBoundingClientRect()
  const before = kgToWorld(event.clientX - rect.left, event.clientY - rect.top)
  const factor = event.deltaY < 0 ? 1.08 : 0.92
  kgScale = Math.max(0.35, Math.min(2.8, kgScale * factor))
  kgOffsetX = event.clientX - rect.left - before.x * kgScale
  kgOffsetY = event.clientY - rect.top - before.y * kgScale
  drawKgCanvas()
}

// ─── Worker ───────────────────────────────────────────────────────────────────

function initWorker() {
  worker = new Worker(new URL('./workers/paperInferenceWorker.ts', import.meta.url), {
    type: 'module',
  })

  worker.addEventListener('message', (e: MessageEvent) => {
    const { type, payload } = e.data
    if (type === 'PROGRESS') {
      extractionStage.value = payload.stage
      extractionProgress.value = payload.percent
      if (payload.stage === 'Models ready') workerReady.value = true
    } else if (type === 'RESULT') {
      handleInferenceResult(payload as EvidencePayload)
    } else if (type === 'ERROR') {
      handleExtractionError(payload.message)
    }
  })

  worker.addEventListener('error', (e) => {
    workerError.value = e.message
    handleExtractionError(`Worker error: ${e.message}`)
  })

  // Pre-warm: load ONNX models into browser cache on startup
  worker.postMessage({ type: 'INIT' })
}

// ─── File Handling ────────────────────────────────────────────────────────────

const handleDrop = (e: DragEvent) => {
  e.preventDefault()
  const file = e.dataTransfer?.files[0]
  if (file) processFile(file)
}

const handleFileSelect = (e: Event) => {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (file) processFile(file)
}

const triggerFileInput = () => fileInput.value?.click()

async function processFile(file: File) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    pushMessage('assistant', 'Please upload a PDF file.')
    return
  }
  paperTitle.value = file.name.replace(/\.pdf$/i, '')
  await runInference(file)
}

// ─── Inference ────────────────────────────────────────────────────────────────

async function runInference(file: File) {
  if (!worker) return

  isExtracting.value = true
  extractionProgress.value = 0
  extractionStage.value = 'Reading PDF…'
  evidencePayload.value = null
  activeSection.value = null
  contextChunks.value = []
  queryKeywords.value = []
  termExplanations.value = []
  kgSemanticEdges.value = []
  if (activeRightView.value === 'graph') activeRightView.value = 'chat'

  try {
    const buffer = await file.arrayBuffer()
    worker.postMessage(
      { type: 'INFER', payload: { pdf: buffer, filename: file.name } },
      [buffer]
    )
  } catch (err: any) {
    handleExtractionError(err?.message ?? 'Failed to read PDF')
  }
}

function handleInferenceResult(payload: EvidencePayload) {
  evidencePayload.value = payload
  totalSentences.value = payload.total_sentences
  sectionsFound.value = payload.sections_found
  activeSection.value = payload.sections_found[0] ?? null
  extractionProgress.value = 100
  isExtracting.value = false

  const sectionList = payload.sections_found.map(s => SECTION_DISPLAY[s] ?? s).join(', ')
  pushMessage(
    'assistant',
    `**"${paperTitle.value}"** analyzed\n\nExtracted **${payload.units.length} concept units** across ${payload.sections_found.length} sections (${sectionList}) from **${payload.total_sentences} sentences** — all computed locally in your browser.\n\nWhat would you like to know?`
  )
}

function handleExtractionError(msg: string) {
  isExtracting.value = false
  extractionProgress.value = 0
  pushMessage('assistant', `${msg}`)
}

// ─── Chat ─────────────────────────────────────────────────────────────────────

function pushMessage(role: 'user' | 'assistant', content: string) {
  messages.value.push({ role, content })
}

const scrollToBottom = async () => {
  await nextTick()
  if (chatContainer.value)
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
}

watch(
  [() => messages.value.length, isGenerating],
  () => scrollToBottom(),
  { deep: true, flush: 'post' }
)

watch(
  [activeRightView, termExplanations],
  async () => {
    if (activeRightView.value === 'graph' && !hasBagOfWordGraph.value) {
      activeRightView.value = 'chat'
      return
    }
    if (activeRightView.value === 'graph' && termExplanations.value.length) {
      await nextTick()
      setupKgCanvas()
    }
  },
  { deep: true, flush: 'post' }
)

async function sendMessage() {
  if (!currentInput.value.trim() || isGenerating.value || !hasPaper.value) return

  const userMsg = currentInput.value.trim()
  pushMessage('user', userMsg)
  currentInput.value = ''
  isGenerating.value = true

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: messages.value,
        evidence_units: evidencePayload.value!.units,
        section_notes: evidencePayload.value!.section_notes,
        title: evidencePayload.value!.title,
      }),
      signal: AbortSignal.timeout(60_000),
    })

    const data = await res.json()
    if (!res.ok) throw new Error(data.error || `Server error ${res.status}`)

    pushMessage('assistant', data.reply)

    // Update context visualization
    if (data.context_chunks?.length) {
      contextChunks.value = data.context_chunks as ContextChunk[]
      queryKeywords.value = data.query_keywords ?? []
      termExplanations.value = data.knowledge_graph?.terms ?? data.term_explanations ?? []
      kgSemanticEdges.value = data.knowledge_graph?.edges ?? []
      if (termExplanations.value.length) selectBowTerm(termExplanations.value[0])
      lastQuery.value = userMsg
      isContextOpen.value = true
    }
  } catch (err: any) {
    const msg = err?.name === 'TimeoutError'
      ? 'Request timed out. Please try again.'
      : err?.message || 'An error occurred.'
    pushMessage('assistant', msg)
  } finally {
    isGenerating.value = false
  }
}

function handleCvpWheel(e: WheelEvent) {
  const target = e.currentTarget as HTMLElement
  target.scrollLeft += e.deltaY
}

const handleKeydown = (e: KeyboardEvent) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

// ─── Lifecycle ────────────────────────────────────────────────────────────────

onMounted(() => {
  initWorker()
  window.addEventListener('resize', resizeKgCanvas)
})

onBeforeUnmount(() => {
  cancelAnimationFrame(kgAnimationFrame)
  window.removeEventListener('resize', resizeKgCanvas)
  worker?.terminate()
})
</script>

<template>
  <div class="layout">
    <!-- Model Loading Overlay -->
    <transition name="fade">
      <div class="model-loading-overlay" v-if="!workerReady && !workerError">
        <div class="model-loading-modal">
          <div class="model-loading-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="url(#loading-grad)" stroke-width="1.5">
              <defs>
                <linearGradient id="loading-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stop-color="#3b82f6" />
                  <stop offset="50%" stop-color="#8b5cf6" />
                  <stop offset="100%" stop-color="#ec4899" />
                </linearGradient>
              </defs>
              <path stroke-linecap="round" stroke-linejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
          </div>
          <h3>Initializing AI</h3>
          <p>{{ extractionStage || 'Downloading ONNX Models...' }}</p>
          <div class="model-loading-track">
            <div class="model-loading-fill" :style="{ width: extractionProgress + '%' }"></div>
          </div>
          <div class="model-loading-pct">{{ extractionProgress }}%</div>
        </div>
      </div>
    </transition>

    <main class="main-content">

      <!-- ═══════════════════════════════════════════════════════════
           LEFT PANEL: Upload + Evidence Concept Units
      ════════════════════════════════════════════════════════════════ -->
      <section class="panel left-panel glass-panel">

        <!-- Brand -->
        <div class="brand-header">
          <div class="brand-icon">🔬</div>
          <div class="brand-text">
            <h1><span class="gradient-text">SciPaper</span> Analyst</h1>
          </div>
        </div>

        <h2 class="panel-title">Document Analysis</h2>

        <!-- Drop Zone -->
        <div
          class="drop-zone"
          :class="{ 'is-loading': isExtracting, 'has-file': !!paperTitle && !isExtracting }"
          @dragover.prevent
          @drop="handleDrop"
          @click="triggerFileInput"
        >
          <input ref="fileInput" type="file" accept=".pdf" @change="handleFileSelect" style="display:none" />



          <div class="drop-text">
            <template v-if="isExtracting">
              <strong>{{ extractionStage || 'Processing…' }}</strong>
              <p>Running SciBERT...</p>
            </template>
            <template v-else-if="paperTitle">
              <strong class="file-title">{{ paperTitle }}</strong>
              <p>{{ totalSentences }} sentences · {{ evidencePayload?.units.length ?? 0 }} concept units</p>
            </template>
            <template v-else>
              <strong>Drag &amp; Drop PDF</strong>
              <p>or click to browse — no upload, fully local</p>
            </template>
          </div>

          <!-- Progress -->
          <div class="progress-track" v-if="isExtracting">
            <div class="progress-fill" :style="{ width: extractionProgress + '%' }"></div>
          </div>
          <div class="progress-pct" v-if="isExtracting">{{ extractionProgress }}%</div>
        </div>

        <!-- Section Tabs -->
        <div class="section-tabs" v-if="sectionTabs.length > 1">
          <button
            class="section-tab"
            :class="{ active: activeSection === null }"
            @click="activeSection = null"
          >All</button>
          <button
            v-for="tab in sectionTabs"
            :key="tab.id"
            class="section-tab"
            :class="{ active: activeSection === tab.id }"
            @click="activeSection = tab.id"
          >{{ tab.label }}</button>
        </div>

        <!-- Evidence Unit Cards -->
        <div class="units-scroll" v-if="evidencePayload && filteredUnits.length > 0">
          <p class="units-heading">Evidence-grounded Concept Units</p>
          <div
            class="unit-card"
            v-for="(unit, idx) in filteredUnits"
            :key="idx"
          >
            <div class="unit-meta">
              <span
                class="role-badge"
                :style="{ background: getRoleColor(unit.role) + '20', color: getRoleColor(unit.role) }"
              >{{ ROLE_BADGES[unit.role]?.label ?? unit.role }}</span>
              <span class="unit-section-tag">{{ SECTION_DISPLAY[unit.section] ?? unit.section }}</span>
              <div class="imp-bar-wrap" :title="`Importance: ${(unit.importance * 100).toFixed(0)}%`">
                <div class="imp-bar" :style="{ width: (unit.importance * 100) + '%' }"></div>
              </div>
            </div>
            <div class="unit-phrase">{{ unit.phrase }}</div>
            <p class="unit-evidence">{{ unit.evidence_sentence }}</p>
          </div>
        </div>

        <!-- Empty state -->
        <div class="empty-hint" v-if="!evidencePayload && !isExtracting">

          <p>Upload a PDF to begin. SciBERT keyword extractor and structure classifier will run entirely in your browser using WebAssembly — no data leaves your device until you start chatting.</p>
        </div>
      </section>

      <!-- ═══════════════════════════════════════════════════════════
           RIGHT PANEL: RAG Context Visualizer + Chat
      ════════════════════════════════════════════════════════════════ -->
      <section class="panel right-panel glass-panel">
        <div class="right-panel-header">
          <h2 class="panel-title">AI Analysis</h2>
          <div class="right-view-tabs" role="tablist" aria-label="AI analysis views">
            <button
              class="right-view-tab"
              :class="{ active: activeRightView === 'chat' }"
              @click="activeRightView = 'chat'"
            >Chat</button>
            <button
              v-if="hasBagOfWordGraph"
              class="right-view-tab"
              :class="{ active: activeRightView === 'graph' }"
              @click="activeRightView = 'graph'"
            >Bag of Word</button>
          </div>
        </div>

        <template v-if="activeRightView === 'chat'">
        <!-- ── RAG Context Visualizer ──────────────────────────────────── -->
        <div class="cvp" v-if="contextChunks.length > 0" :class="{ open: isContextOpen }">
          <!-- Header -->
          <div class="cvp-header" @click="isContextOpen = !isContextOpen">
            <div class="cvp-left">
              <span class="cvp-title">Retrieved Evidence</span>
              <span class="cvp-query" v-if="lastQuery">
                "{{ lastQuery.length > 50 ? lastQuery.slice(0, 50) + '…' : lastQuery }}"
              </span>
              <span class="cvp-badge">{{ contextChunks.length }} passages</span>
            </div>
            <div class="cvp-right">
              <!-- BoW Keyword Pills -->
              <div class="kw-pills" v-if="queryKeywords.length">
                <span class="kw-pill" v-for="kw in queryKeywords.slice(0, 5)" :key="kw">{{ kw }}</span>
              </div>
              <button class="cvp-chevron" :class="{ flipped: !isContextOpen }">
                <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
                  <path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"/>
                </svg>
              </button>
            </div>
          </div>

          <!-- Scrollable Card Row -->
          <Transition name="cvp-slide">
            <div class="cvp-body" v-show="isContextOpen" @wheel.prevent="handleCvpWheel">
              <div
                class="cvp-card"
                v-for="(chunk, idx) in contextChunks"
                :key="idx"
                :style="{ '--accent': getRoleColor(chunk.role ?? '') }"
              >
                <!-- Card header -->
                <div class="cvp-card-top">
                  <span class="cvp-rank">#{{ idx + 1 }}</span>
                  <span class="cvp-sec">{{ chunk.section_label }}</span>

                  <span
                    class="cvp-role"
                    v-if="chunk.role && chunk.role !== 'none'"
                    :style="{ background: getRoleColor(chunk.role) + '22', color: getRoleColor(chunk.role) }"
                  >{{ ROLE_BADGES[chunk.role]?.label ?? chunk.role }}</span>

                  <span class="cvp-phrase-tag" v-if="chunk.phrase">{{ chunk.phrase }}</span>

                  <!-- Score bar -->
                  <div class="cvp-score-row">
                    <div class="cvp-score-track" :title="`Relevance: ${(chunk.score * 100).toFixed(1)}%`">
                      <div class="cvp-score-fill" :style="{ width: Math.min(chunk.score, 1) * 100 + '%' }"></div>
                    </div>
                    <span class="cvp-pct">{{ (chunk.score * 100).toFixed(0) }}%</span>
                    <template v-if="chunk.importance !== null">
                      <span class="cvp-dot">·</span>
                      <span class="cvp-pct imp">imp {{ (chunk.importance * 100).toFixed(0) }}%</span>
                    </template>
                    <template v-if="chunk.evidence_score !== null && chunk.evidence_score > 0.05">
                      <span class="cvp-dot">·</span>
                      <span class="cvp-pct ev">ev {{ (chunk.evidence_score * 100).toFixed(0) }}%</span>
                    </template>
                  </div>
                </div>

                <!-- BoW matched keyword chips -->
                <div class="cvp-kw-row" v-if="chunk.matched_keywords.length">
                  <svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor" style="opacity:.5;flex-shrink:0">
                    <path d="M14 6H2a1 1 0 000 2h12a1 1 0 000-2zM2 4h12a1 1 0 000-2H2a1 1 0 000 2zm12 6H2a1 1 0 000 2h12a1 1 0 000-2z"/>
                  </svg>
                  <span class="kw-chip" v-for="kw in chunk.matched_keywords" :key="kw">{{ kw }}</span>
                </div>

                <!-- Passage with BoW highlight -->
                <p class="cvp-passage" v-html="highlightText(chunk.text, chunk.matched_keywords)"></p>
              </div>
            </div>
          </Transition>
        </div>
        <!-- /RAG Context Visualizer -->

        <!-- Chat Messages -->
        <div class="chat-messages" ref="chatContainer">
          <div
            v-for="(msg, idx) in messages"
            :key="idx"
            class="msg"
            :class="msg.role"
          >
            <div class="avatar">{{ msg.role === 'user' ? 'You' : 'AI' }}</div>
            <div class="bubble" v-html="renderMessage(msg.content)"></div>
          </div>

          <div class="msg assistant typing" v-if="isGenerating">
            <div class="avatar">AI</div>
            <div class="bubble dots">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>

        <!-- Input -->
        <div class="chat-input" :class="{ disabled: !hasPaper }">
          <textarea
            v-model="currentInput"
            :placeholder="hasPaper
              ? 'Ask about methodology, results, contributions… (Enter to send)'
              : 'Upload a PDF to start chatting…'"
            :disabled="!hasPaper || isGenerating"
            @keydown="handleKeydown"
            rows="2"
          ></textarea>
          <button
            class="send-btn"
            :disabled="!currentInput.trim() || isGenerating || !hasPaper"
            @click="sendMessage"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <path d="M5 12h14M12 5l7 7-7 7"/>
            </svg>
          </button>
        </div>
        </template>

        <template v-else>
          <div class="kg-workspace">
            <div class="kg-toolbar">
              <div>
                <p class="kg-eyebrow">Bag of Word</p>
                <h3>Knowledge Graph Based on BoW and Your Question</h3>
              </div>
              <div class="kg-stat-row">
                <span>{{ graphConceptCount }} concepts</span>
                <span>{{ graphCategoryCount }} categories</span>
              </div>
            </div>

            <div class="kg-main">
              <div class="kg-canvas" v-if="termExplanations.length">
                <canvas
                  ref="kgCanvas"
                  class="kg-dynamic-canvas"
                  @pointerdown="handleKgPointerDown"
                  @pointermove="handleKgPointerMove"
                  @pointerup="handleKgPointerUp"
                  @pointerleave="handleKgPointerUp"
                  @wheel.prevent="handleKgWheel"
                ></canvas>
                <div ref="kgTooltip" class="kg-tooltip"></div>
              </div>

              <div class="kg-canvas" v-else>
                <svg viewBox="0 0 920 360" role="img" aria-label="Global concept graph preview">
                  <g v-for="category in GLOBAL_KG_CATEGORIES" :key="category.id">
                    <line
                      v-for="concept in category.concepts"
                      :key="category.id + concept.label"
                      :x1="category.x"
                      :y1="category.y"
                      :x2="concept.x"
                      :y2="concept.y"
                      class="kg-edge"
                    />
                    <circle
                      class="kg-category-node"
                      :cx="category.x"
                      :cy="category.y"
                      r="46"
                      :style="{ '--node-color': category.color }"
                    />
                    <text class="kg-category-label" :x="category.x" :y="category.y - 4">{{ category.label }}</text>
                    <text class="kg-category-sub" :x="category.x" :y="category.y + 15">category</text>

                    <g
                      v-for="concept in category.concepts"
                      :key="concept.label"
                      class="kg-concept-group"
                      @click="selectedKgConcept = {
                        label: concept.label,
                        category: category.label,
                        df: concept.df,
                        tf: concept.tf,
                        confidence: concept.confidence,
                        aliases: concept.aliases,
                        wikidata: concept.wikidata,
                        wikidataUrl: getWikidataUrl(concept.wikidata)
                      }"
                    >
                      <circle
                        class="kg-concept-node"
                        :cx="concept.x"
                        :cy="concept.y"
                        :r="20 + concept.confidence * 8"
                        :style="{ '--node-color': category.color }"
                      />
                      <text class="kg-concept-label" :x="concept.x" :y="concept.y + 42">{{ concept.label }}</text>
                    </g>
                  </g>
                </svg>
              </div>

              <aside class="kg-detail">
                <p class="kg-eyebrow">Selected Concept</p>
                <h3>{{ selectedKgConcept.label }}</h3>
                <div class="kg-detail-grid">
                  <span>Category</span><strong>{{ selectedKgConcept.category }}</strong>
                  <span>Document Frequency</span><strong>{{ selectedKgConcept.df }}</strong>
                  <span>Total Frequency</span><strong>{{ selectedKgConcept.tf }}</strong>
                  <span>Wikidata</span>
                  <strong>
                    <a
                      v-if="selectedKgConcept.wikidataUrl"
                      class="kg-wikidata-link"
                      :href="selectedKgConcept.wikidataUrl"
                      target="_blank"
                      rel="noopener noreferrer"
                    >{{ selectedKgConcept.wikidata }}</a>
                    <span v-else>{{ selectedKgConcept.wikidata }}</span>
                  </strong>
                </div>
                <div class="kg-alias-box">
                  <span>{{ termExplanations.length ? 'Matched Alias' : 'Aliases' }}</span>
                  <p>{{ selectedKgConcept.aliases }}</p>
                </div>
              </aside>
            </div>
          </div>
        </template>
      </section>
    </main>
  </div>
</template>

<style scoped>
/* ═══════════════════════════════════════════════════════════ Layout */
.fade-enter-active, .fade-leave-active { transition: opacity 0.4s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

.model-loading-overlay {
  position: fixed;
  top: 0; left: 0; width: 100vw; height: 100vh;
  background: rgba(15, 23, 42, 0.6);
  backdrop-filter: blur(12px);
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
}

.model-loading-modal {
  background: rgba(255, 255, 255, 0.9);
  padding: 3rem 2.5rem;
  border-radius: 24px;
  box-shadow: 0 20px 40px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.6);
  text-align: center;
  width: 380px;
  max-width: 90vw;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.2rem;
  border: 1px solid rgba(255,255,255,0.5);
}

.model-loading-icon svg {
  width: 64px;
  height: 64px;
  animation: float-icon 3s ease-in-out infinite;
}

@keyframes float-icon {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-8px); }
}

.model-loading-modal h3 {
  margin: 0;
  font-size: 1.35rem;
  font-weight: 700;
  background: linear-gradient(135deg, #1e293b, #334155);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.model-loading-modal p {
  margin: 0;
  font-size: 0.9rem;
  color: #64748b;
}

.model-loading-track {
  width: 100%;
  height: 8px;
  background: #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: inset 0 1px 2px rgba(0,0,0,0.06);
}

.model-loading-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
  background-size: 200% 100%;
  border-radius: 8px;
  transition: width 0.3s ease;
  animation: gradient-shift 2s linear infinite;
}

.model-loading-pct {
  font-size: 0.85rem;
  font-weight: 600;
  color: #94a3b8;
  font-family: monospace;
}

@keyframes gradient-shift {
  0% { background-position: 100% 0; }
  100% { background-position: -100% 0; }
}

.layout {
  width: 100%;
  max-width: 1500px;
  height: 100vh;
  display: flex;
  flex-direction: column;
  padding: 1.25rem;
}

.main-content {
  display: flex;
  gap: 1.25rem;
  flex: 1;
  min-height: 0;
  animation: fadeIn 0.4s ease-out both;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

.panel {
  display: flex;
  flex-direction: column;
  padding: 1.4rem;
  gap: 1.1rem;
  overflow: hidden;
}

.left-panel  { flex: 0 0 400px; }
.right-panel { flex: 1; }

/* ═══════════════════════════════════════════════════════════ Brand */
.brand-header {
  display: flex;
  align-items: center;
  gap: 0.85rem;
  padding-bottom: 1.1rem;
  border-bottom: 1px solid var(--glass-border);
}

.brand-icon { font-size: 2rem; }

.brand-text {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.brand-text h1 {
  font-size: 1.6rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin: 0;
}

.gradient-text {
  background: linear-gradient(135deg, var(--primary), var(--accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.edge-badge {
  font-size: 0.72rem;
  font-weight: 700;
  padding: 2px 10px;
  border-radius: 20px;
  background: rgba(16, 185, 129, 0.12);
  color: #059669;
  border: 1px solid rgba(16, 185, 129, 0.3);
  font-family: Arial, sans-serif;
  letter-spacing: 0.03em;
  width: fit-content;
}

.panel-title {
  font-size: 0.88rem;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.07em;
}

.right-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  flex-shrink: 0;
}

.right-view-tabs {
  display: flex;
  gap: 0.3rem;
  padding: 0.25rem;
  background: rgba(255,255,255,0.72);
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  flex-shrink: 0;
}

.right-view-tab {
  min-width: 82px;
  padding: 0.45rem 0.75rem;
  border-radius: 7px;
  background: transparent;
  color: var(--text-muted);
  font-size: 0.78rem;
  font-weight: 700;
  font-family: Arial, sans-serif;
}

.right-view-tab:hover {
  background: rgba(99,102,241,0.07);
  color: var(--primary);
}

.right-view-tab.active {
  background: var(--primary);
  color: #fff;
  box-shadow: 0 3px 10px rgba(79,70,229,0.22);
}

/* ═══════════════════════════════════════════════════════════ Drop Zone */
.drop-zone {
  border: 2px dashed #cbd5e1;
  border-radius: 14px;
  min-height: 130px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  padding: 1.4rem;
  cursor: pointer;
  transition: all 0.2s ease;
  background: rgba(255,255,255,0.5);
  text-align: center;
  flex-shrink: 0;
}

.drop-zone:hover    { border-color: var(--primary); background: rgba(99,102,241,0.04); transform: translateY(-1px); }
.drop-zone.has-file { border-color: #10b981; background: rgba(16,185,129,0.04); }
.drop-zone.is-loading { border-color: var(--primary); animation: pulse-border 1.6s infinite; }

@keyframes pulse-border {
  0%, 100% { border-color: var(--primary); }
  50%       { border-color: var(--accent); }
}

.drop-icon { font-size: 2rem; }

.drop-text strong { font-size: 1rem; color: var(--text-main); display: block; margin-bottom: 4px; }
.drop-text p      { font-size: 0.9rem; color: var(--text-muted); }

.file-title { color: var(--primary) !important; word-break: break-all; }

.progress-track {
  width: 80%;
  height: 4px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--primary), var(--accent));
  border-radius: 4px;
  transition: width 0.4s ease;
}

.progress-pct {
  font-size: 0.72rem;
  color: var(--text-muted);
  font-family: Arial, sans-serif;
}

/* ═══════════════════════════════════════════════════════════ Section Tabs */
.section-tabs {
  display: flex;
  gap: 0.35rem;
  flex-wrap: wrap;
  flex-shrink: 0;
}

.section-tab {
  font-size: 0.78rem;
  font-weight: 600;
  padding: 4px 12px;
  border-radius: 20px;
  border: 1px solid #e2e8f0;
  background: #fff;
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.2s;
  font-family: Arial, sans-serif;
}

.section-tab:hover  { border-color: var(--primary); color: var(--primary); }
.section-tab.active { background: var(--primary); color: #fff; border-color: var(--primary); }

/* ═══════════════════════════════════════════════════════════ Unit Cards */
.units-scroll {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.units-heading {
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.unit-card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 0.7rem 0.85rem;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  transition: all 0.2s;
  animation: cardIn 0.25s ease-out both;
}

.unit-card:hover { border-color: var(--primary); box-shadow: 0 4px 12px rgba(99,102,241,0.1); transform: translateY(-1px); }

@keyframes cardIn {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: translateY(0); }
}

.unit-meta { display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.35rem; flex-wrap: wrap; }

.role-badge {
  font-size: 0.7rem;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 10px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-family: Arial, sans-serif;
}

.unit-section-tag {
  font-size: 0.7rem;
  color: var(--text-muted);
  font-style: italic;
}

.imp-bar-wrap {
  margin-left: auto;
  width: 38px;
  height: 3px;
  background: #e2e8f0;
  border-radius: 3px;
  overflow: hidden;
}

.imp-bar {
  height: 100%;
  background: linear-gradient(90deg, var(--primary), var(--accent));
  border-radius: 3px;
}

.unit-phrase {
  font-size: 0.93rem;
  font-weight: 600;
  color: var(--text-main);
  margin-bottom: 0.28rem;
}

.unit-evidence {
  font-size: 0.84rem;
  color: var(--text-muted);
  line-height: 1.5;
  margin: 0;
  max-height: 4.5em; /* Approximately 3 lines */
  overflow-y: auto;
  overscroll-behavior-y: contain;
  display: block;
}

.unit-evidence::-webkit-scrollbar {
  width: 4px;
}
.unit-evidence::-webkit-scrollbar-track {
  background: transparent;
}
.unit-evidence::-webkit-scrollbar-thumb {
  background: rgba(148, 163, 184, 0.4);
  border-radius: 4px;
}
.unit-evidence::-webkit-scrollbar-thumb:hover {
  background: rgba(148, 163, 184, 0.6);
}

.more-hint {
  text-align: center;
  font-size: 0.82rem;
  color: var(--text-muted);
  font-style: italic;
  padding: 0.4rem;
}

.empty-hint {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  text-align: center;
  padding: 1rem;
}

.empty-icon { font-size: 3rem; }

.empty-hint p {
  font-size: 0.95rem;
  color: var(--text-muted);
  line-height: 1.7;
  max-width: 280px;
}

/* ═══════════════════════════════════════════════════════════ RAG Context Visualizer */

.cvp {
  flex-shrink: 0;
  border-radius: 12px;
  border: 1px solid rgba(99,102,241,0.18);
  background: rgba(248,250,252,0.75);
  backdrop-filter: blur(8px);
  overflow: hidden;
  animation: slideDown 0.3s ease-out both;
}

.cvp.open { border-color: rgba(99,102,241,0.32); box-shadow: 0 4px 18px rgba(99,102,241,0.07); }

@keyframes slideDown {
  from { opacity: 0; transform: translateY(-6px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* CVP Header */
.cvp-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.55rem 0.85rem;
  cursor: pointer;
  user-select: none;
  transition: background 0.2s;
  border-bottom: 1px solid transparent;
  gap: 0.6rem;
}

.cvp.open .cvp-header { border-bottom-color: rgba(99,102,241,0.1); background: rgba(99,102,241,0.025); }
.cvp-header:hover { background: rgba(99,102,241,0.04); }
.cvp-left  { display: flex; align-items: center; gap: 0.45rem; flex: 1; min-width: 0; }
.cvp-right { display: flex; align-items: center; gap: 0.4rem; flex-shrink: 0; }

.cvp-icon { font-size: 0.95rem; }

.cvp-title {
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--primary);
  text-transform: uppercase;
  letter-spacing: 0.07em;
  font-family: Arial, sans-serif;
  white-space: nowrap;
}

.cvp-query {
  font-size: 0.75rem;
  color: var(--text-muted);
  font-style: italic;
  background: rgba(148,163,184,0.1);
  padding: 1px 7px;
  border-radius: 20px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 180px;
}

.cvp-badge {
  font-size: 0.68rem;
  font-weight: 700;
  padding: 1px 6px;
  background: rgba(99,102,241,0.1);
  color: var(--primary);
  border-radius: 20px;
  font-family: Arial, sans-serif;
  flex-shrink: 0;
}

.kw-pills { display: flex; gap: 0.25rem; }

.kw-pill {
  font-size: 0.67rem;
  font-weight: 600;
  padding: 1px 6px;
  background: rgba(245,158,11,0.14);
  color: #d97706;
  border: 1px solid rgba(245,158,11,0.28);
  border-radius: 20px;
  font-family: Arial, sans-serif;
  white-space: nowrap;
}

.cvp-chevron {
  width: 24px; height: 24px;
  border-radius: 6px;
  background: rgba(99,102,241,0.08);
  color: var(--primary);
  display: flex; align-items: center; justify-content: center;
  border: none; cursor: pointer;
  transition: background 0.2s;
}
.cvp-chevron:hover { background: rgba(99,102,241,0.14); }
.cvp-chevron.flipped svg { transform: rotate(180deg); }
.cvp-chevron svg { transition: transform 0.28s ease; }

/* CVP Body (horizontal scroll) */
.cvp-body {
  display: flex;
  gap: 0.55rem;
  padding: 0.6rem 0.85rem;
  overflow-x: auto;
  scroll-snap-type: x mandatory;
}

.cvp-body::-webkit-scrollbar { height: 8px; }
.cvp-body::-webkit-scrollbar-track { background: transparent; }
.cvp-body::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.4); border-radius: 4px; }
.cvp-body::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.6); }

/* Slide animation */
.cvp-slide-enter-active { transition: all 0.26s cubic-bezier(0.4,0,0.2,1); }
.cvp-slide-leave-active { transition: all 0.2s cubic-bezier(0.4,0,0.2,1); }
.cvp-slide-enter-from, .cvp-slide-leave-to { opacity: 0; max-height: 0; }
.cvp-slide-enter-to, .cvp-slide-leave-from { opacity: 1; max-height: 500px; }

/* CVP Evidence Cards */
.cvp-card {
  flex: 0 0 296px;
  scroll-snap-align: start;
  background: #fff;
  border-radius: 9px;
  border: 1px solid #e8edf4;
  border-left: 3px solid var(--accent, #6366f1);
  padding: 0.6rem 0.75rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04);
  transition: box-shadow 0.2s, transform 0.2s;
  animation: cardPop 0.28s ease-out both;
}

.cvp-card:nth-child(1) { animation-delay: 0s;    }
.cvp-card:nth-child(2) { animation-delay: 0.05s; }
.cvp-card:nth-child(3) { animation-delay: 0.10s; }
.cvp-card:nth-child(4) { animation-delay: 0.15s; }
.cvp-card:nth-child(5) { animation-delay: 0.20s; }

@keyframes cardPop {
  from { opacity: 0; transform: translateY(5px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}

.cvp-card:hover { box-shadow: 0 5px 16px rgba(0,0,0,0.08); transform: translateY(-2px); }

.cvp-card-top { display: flex; align-items: center; gap: 0.3rem; flex-wrap: wrap; }

.cvp-rank {
  font-size: 0.66rem;
  font-weight: 800;
  color: var(--accent, #6366f1);
  background: rgba(99,102,241,0.07);
  padding: 1px 5px;
  border-radius: 5px;
  font-family: Arial, sans-serif;
  flex-shrink: 0;
}

.cvp-sec { font-size: 0.68rem; color: var(--text-muted); font-style: italic; }

.cvp-role {
  font-size: 0.65rem;
  font-weight: 700;
  padding: 1px 5px;
  border-radius: 8px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-family: Arial, sans-serif;
  flex-shrink: 0;
}

.cvp-phrase-tag {
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--text-main);
  background: rgba(99,102,241,0.07);
  padding: 1px 5px;
  border-radius: 5px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100px;
}

/* Score row */
.cvp-score-row { display: flex; align-items: center; gap: 0.25rem; margin-left: auto; flex-shrink: 0; }

.cvp-score-track {
  width: 40px;
  height: 3px;
  background: #e2e8f0;
  border-radius: 3px;
  overflow: hidden;
}

.cvp-score-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--primary), var(--accent));
  border-radius: 3px;
  transition: width 0.5s cubic-bezier(0.34,1.56,0.64,1);
}

.cvp-pct     { font-size: 0.62rem; font-weight: 700; color: var(--text-muted); white-space: nowrap; font-family: Arial, sans-serif; }
.cvp-pct.imp { color: #7c3aed; }
.cvp-pct.ev  { color: #059669; }
.cvp-dot     { color: #d1d5db; font-size: 0.85rem; }

/* Keyword chip row */
.cvp-kw-row { display: flex; align-items: center; gap: 0.25rem; flex-wrap: wrap; }

.kw-chip {
  font-size: 0.63rem;
  font-weight: 600;
  padding: 1px 5px;
  background: rgba(245,158,11,0.1);
  color: #b45309;
  border: 1px solid rgba(245,158,11,0.22);
  border-radius: 8px;
  font-family: Arial, sans-serif;
  white-space: nowrap;
}

/* Passage with BoW highlights */
.cvp-passage {
  font-size: 0.8rem;
  line-height: 1.65;
  color: var(--text-muted);
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
  margin: 0;
  word-break: break-word;
}

/* BoW highlight mark — gold glow */
:deep(.bow-hit) {
  background: linear-gradient(120deg, rgba(251,191,36,0.42) 0%, rgba(245,158,11,0.3) 100%);
  color: #92400e;
  border-radius: 3px;
  padding: 0 2px;
  font-weight: 700;
  border-bottom: 1.5px solid rgba(245,158,11,0.55);
  transition: background 0.18s;
}

.cvp-passage:hover :deep(.bow-hit) {
  background: linear-gradient(120deg, rgba(251,191,36,0.62) 0%, rgba(245,158,11,0.46) 100%);
}

/* ═══════════════════════════════════════════════════════════ Evidence Workspace */
.evidence-workspace,
.kg-workspace {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
  animation: fadeIn 0.28s ease-out both;
}

.evidence-summary,
.kg-toolbar {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.85rem 1rem;
  border-radius: 12px;
  background: rgba(255,255,255,0.72);
  border: 1px solid #e2e8f0;
}

.kg-eyebrow {
  margin: 0 0 0.25rem;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--primary);
  font-family: Arial, sans-serif;
}

.evidence-summary h3,
.kg-toolbar h3,
.kg-detail h3,
.kg-empty-state h3 {
  margin: 0;
  font-size: 1rem;
  color: var(--text-main);
}

.evidence-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
  padding-right: 0.35rem;
}

.evidence-row {
  border-radius: 10px;
  border: 1px solid #e2e8f0;
  border-left: 3px solid var(--accent, #6366f1);
  background: rgba(255,255,255,0.82);
  padding: 0.78rem 0.9rem;
  box-shadow: 0 2px 8px rgba(15,23,42,0.04);
}

.evidence-row-top {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  flex-wrap: wrap;
  margin-bottom: 0.45rem;
}

.evidence-score {
  margin-left: auto;
  font-size: 0.68rem;
  font-weight: 800;
  color: var(--primary);
  font-family: Arial, sans-serif;
}

.evidence-passage {
  margin: 0;
  color: var(--text-muted);
  font-size: 0.9rem;
  line-height: 1.65;
}

.kg-empty-state {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 2rem;
  border-radius: 12px;
  border: 1px dashed #cbd5e1;
  background: rgba(255,255,255,0.55);
}

.kg-empty-state p:last-child {
  max-width: 420px;
  margin: 0.6rem 0 0;
  color: var(--text-muted);
  line-height: 1.7;
}

/* ═══════════════════════════════════════════════════════════ Knowledge Graph */
.kg-stat-row {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.kg-stat-row span {
  font-size: 0.7rem;
  font-weight: 800;
  color: var(--primary);
  background: rgba(99,102,241,0.09);
  border: 1px solid rgba(99,102,241,0.16);
  border-radius: 20px;
  padding: 0.22rem 0.55rem;
  font-family: Arial, sans-serif;
}

.kg-main {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 260px;
  gap: 0.85rem;
}

.kg-canvas,
.kg-detail {
  min-height: 0;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  background: rgba(255,255,255,0.72);
  box-shadow: 0 2px 8px rgba(15,23,42,0.04);
}

.kg-canvas {
  overflow: hidden;
  position: relative;
}

.kg-dynamic-canvas {
  width: 100%;
  height: 100%;
  min-height: 410px;
  display: block;
  cursor: grab;
  background: #fbfcfd;
}

.kg-dynamic-canvas:active {
  cursor: grabbing;
}

.kg-tooltip {
  position: absolute;
  z-index: 3;
  display: none;
  max-width: 260px;
  transform: translate(12px, 12px);
  pointer-events: none;
  padding: 0.45rem 0.55rem;
  border-radius: 7px;
  background: rgba(15, 23, 42, 0.92);
  color: #fff;
  font-size: 0.72rem;
  line-height: 1.35;
  box-shadow: 0 8px 18px rgba(15,23,42,0.18);
}

.kg-canvas svg {
  width: 100%;
  height: 100%;
  min-height: 410px;
  display: block;
}

.kg-edge {
  stroke: #cbd5e1;
  stroke-width: 2;
  stroke-linecap: round;
}

.kg-paper-link {
  stroke: color-mix(in srgb, var(--node-color) 46%, #cbd5e1);
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-dasharray: 5 5;
  opacity: 0.75;
}

.kg-category-node,
.kg-concept-node {
  fill: color-mix(in srgb, var(--node-color) 14%, #ffffff);
  stroke: var(--node-color);
  stroke-width: 2;
}

.kg-category-node {
  filter: drop-shadow(0 7px 12px rgba(15,23,42,0.08));
}

.kg-concept-group {
  cursor: pointer;
}

.kg-concept-group:hover .kg-concept-node {
  fill: color-mix(in srgb, var(--node-color) 24%, #ffffff);
  stroke-width: 3;
}

.kg-concept-group.active .kg-concept-node {
  fill: color-mix(in srgb, var(--node-color) 30%, #ffffff);
  stroke-width: 3;
}

.kg-category-label,
.kg-category-sub,
.kg-concept-label,
.kg-concept-score {
  text-anchor: middle;
  dominant-baseline: middle;
  pointer-events: none;
  font-family: Arial, sans-serif;
}

.kg-category-label {
  font-size: 0.78rem;
  font-weight: 800;
  fill: var(--text-main);
}

.kg-category-sub {
  font-size: 0.62rem;
  font-weight: 700;
  fill: var(--text-muted);
}

.kg-concept-label {
  font-size: 0.72rem;
  font-weight: 700;
  fill: var(--text-main);
}

.kg-concept-score {
  font-size: 0.62rem;
  font-weight: 800;
  fill: var(--text-muted);
}

.kg-detail {
  padding: 1rem;
  overflow-y: auto;
}

.kg-detail-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0.35rem;
  margin-top: 1rem;
}

.kg-detail-grid span,
.kg-alias-box span {
  font-size: 0.67rem;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
  font-family: Arial, sans-serif;
}

.kg-detail-grid strong {
  font-size: 0.9rem;
  color: var(--text-main);
  margin-bottom: 0.35rem;
  word-break: break-word;
}

.kg-wikidata-link {
  color: #2563eb;
  text-decoration: none;
  font-weight: 800;
}

.kg-wikidata-link:hover {
  text-decoration: underline;
}

.kg-alias-box {
  margin-top: 1rem;
  padding: 0.75rem;
  border-radius: 9px;
  background: rgba(99,102,241,0.06);
  border: 1px solid rgba(99,102,241,0.12);
}

.kg-alias-box p {
  margin: 0.35rem 0 0;
  color: var(--text-muted);
  font-size: 0.86rem;
  line-height: 1.6;
}

/* ═══════════════════════════════════════════════════════════ Chat */
.chat-messages {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  padding-right: 0.5rem;
  min-height: 0;
}

.msg { display: flex; gap: 0.75rem; align-items: flex-end; }
.msg.user { flex-direction: row-reverse; }

.avatar {
  width: 40px; height: 40px;
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.88rem; font-weight: 700;
  font-family: Arial, sans-serif;
  flex-shrink: 0;
}

.msg.user .avatar      { background: linear-gradient(135deg, var(--primary), var(--accent)); color: #fff; }
.msg.assistant .avatar { background: #fff; border: 1px solid #e2e8f0; color: var(--primary); }

.bubble {
  background: #fff;
  padding: 0.9rem 1.1rem;
  border-radius: 16px;
  max-width: 95%;
  line-height: 1.65;
  font-size: 1rem;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04);
  border: 1px solid #e8edf4;
  word-break: break-word;
}

.msg.user .bubble {
  background: linear-gradient(135deg, var(--primary), #6366f1cc);
  color: #fff;
  border-color: transparent;
  border-bottom-right-radius: 4px;
}

.msg.assistant .bubble { border-bottom-left-radius: 4px; }

.bubble :deep(code) {
  background: rgba(99,102,241,0.08);
  color: var(--primary);
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 0.88em;
  font-family: 'Fira Code', monospace;
}

.bubble :deep(p) { margin: 0.2rem 0; }
.bubble :deep(ul), .bubble :deep(ol) {
  padding-left: 1.6rem;
  margin: 0.4rem 0;
}
.bubble :deep(li) { margin-bottom: 0.2rem; }

/* Typing indicator */
.dots { display: flex; align-items: center; gap: 5px; min-height: 36px; }
.dots span {
  width: 7px; height: 7px;
  background: #94a3b8;
  border-radius: 50%;
  animation: bounce 1.3s infinite ease-in-out both;
}
.dots span:nth-child(1) { animation-delay: -0.32s; }
.dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0.5); opacity: 0.4; }
  40%            { transform: scale(1);   opacity: 1; }
}

/* Chat input */
.chat-input {
  display: flex;
  gap: 0.6rem;
  align-items: flex-end;
  background: #fff;
  padding: 0.5rem;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.03);
  transition: border-color 0.2s, box-shadow 0.2s;
  flex-shrink: 0;
}

.chat-input:focus-within {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(99,102,241,0.08);
}

.chat-input.disabled { opacity: 0.55; }

textarea {
  flex: 1;
  background: transparent;
  border: none;
  resize: none;
  padding: 8px 10px;
  font-size: 1rem;
  line-height: 1.5;
  min-height: 44px;
  max-height: 160px;
  color: var(--text-main);
  font-family: inherit;
}
textarea:focus    { outline: none; }
textarea:disabled { cursor: not-allowed; color: var(--text-muted); }

.send-btn {
  background: linear-gradient(135deg, var(--primary), var(--accent));
  color: #fff;
  width: 40px; height: 40px;
  border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  transition: all 0.2s;
}

.send-btn:hover:not(:disabled) {
  opacity: 0.88;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(99,102,241,0.3);
}

.send-btn:disabled {
  background: #e2e8f0;
  color: #94a3b8;
  cursor: not-allowed;
}

.send-btn svg { width: 18px; height: 18px; }

/* Scrollbars */
.chat-messages::-webkit-scrollbar,
.units-scroll::-webkit-scrollbar { width: 4px; }
.chat-messages::-webkit-scrollbar-thumb,
.units-scroll::-webkit-scrollbar-thumb {
  background: rgba(148,163,184,0.32);
  border-radius: 4px;
}
</style>
