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

interface ConceptUnit {
  section: string
  phrase: string
  role: string
  evidence_sentence: string
  importance: number
  sentence_index: number
  boundary_score: number
  evidence_score: number
  role_score: number
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
  source: 'edge'
}

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
const isContextOpen = ref(true)
const lastQuery = ref('')

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
})

onBeforeUnmount(() => {
  worker?.terminate()
})
</script>

<template>
  <div class="layout">
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
        <h2 class="panel-title">AI Analysis Chat</h2>

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
      </section>
    </main>
  </div>
</template>

<style scoped>
/* ═══════════════════════════════════════════════════════════ Layout */
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
