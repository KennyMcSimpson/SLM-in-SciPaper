/**
 * paperInferenceWorker.ts
 * ========================
 * Step 2 – Client-Edge-Cloud: Browser-side PDF parsing + ONNX inference.
 *
 * This Web Worker runs in an isolated thread. It:
 *   1. Receives a PDF ArrayBuffer from the main thread.
 *   2. Uses PDF.js to extract plain text (no server upload needed).
 *   3. Loads ONNX Runtime Web and runs the INT8-quantized SciBERT models.
 *   4. Builds Evidence-grounded Concept Unit JSON cards (mirroring infer_paper_card.py).
 *   5. Posts structured results back to the main thread.
 *
 * Communication protocol (postMessage):
 *
 *  Main → Worker:
 *    { type: 'INIT' }                              – preload models into IndexedDB cache
 *    { type: 'INFER', payload: { pdf: ArrayBuffer, filename: string } }
 *
 *  Worker → Main:
 *    { type: 'PROGRESS', payload: { stage: string, percent: number } }
 *    { type: 'RESULT',   payload: EvidencePayload }
 *    { type: 'ERROR',    payload: { message: string } }
 *
 * Model files must be placed in /public/models/:
 *    keyword_extractor_int8.onnx
 *    keyword_vocab.json
 *    structure_model_int8.onnx
 *    structure_vocab.json
 *    manifest.json
 */

import * as ort from 'onnxruntime-web';
import * as pdfjsLib from 'pdfjs-dist';

// ─── Type declarations ────────────────────────────────────────────────────────

interface WorkerMessage {
  type: 'INIT' | 'INFER'
  payload?: {
    pdf?: ArrayBuffer
    filename?: string
  }
}

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
  chunk_text: string[]   // ← plain text chunks for backend RAG fallback
}

interface SentenceRecord {
  text: string
  section: string
  sentence_index: number
}

// ─── Constants ────────────────────────────────────────────────────────────────

const MODEL_BASE = '/models/'
const MAX_SEQ_LEN = 512
const MAX_SENTENCES_KW = 16
const MAX_SENTENCES_ST = 48
const CHUNK_SIZE = 800    // chars, mirrors backend chunk_text()
const CHUNK_OVERLAP = 100

const CANONICAL_SECTIONS = ['intro', 'related_work', 'method', 'experiment', 'conclusion']
const ROLE_LABELS = [
  'none', 'background', 'problem', 'motivation', 'objective',
  'prior_work', 'limitation', 'gap', 'comparison',
  'core_method', 'component', 'mechanism', 'process',
  'dataset', 'metric', 'baseline', 'result', 'ablation',
  'contribution', 'finding', 'future_work',
]

const SECTION_ALIASES: Record<string, string> = {
  abstract: 'intro', introduction: 'intro', intro: 'intro', background: 'intro',
  'related work': 'related_work', related_work: 'related_work', 'prior work': 'related_work',
  methods: 'method', method: 'method', methodology: 'method',
  experiments: 'experiment', experiment: 'experiment', evaluation: 'experiment', results: 'experiment',
  conclusion: 'conclusion', conclusions: 'conclusion',
}

// ─── Tokenizer (WordPiece, minimal JS implementation) ─────────────────────────

class WordPieceTokenizer {
  private vocab: Map<string, number>
  private vocabArr: string[]
  private unkId: number
  private clsId: number
  private sepId: number
  private padId: number

  constructor(vocab: Record<string, number>) {
    this.vocab = new Map(Object.entries(vocab).map(([k, v]) => [k, v as number]))
    this.vocabArr = Array(this.vocab.size)
    for (const [token, id] of this.vocab) this.vocabArr[id] = token
    this.unkId = this.vocab.get('[UNK]') ?? 100
    this.clsId = this.vocab.get('[CLS]') ?? 101
    this.sepId = this.vocab.get('[SEP]') ?? 102
    this.padId = this.vocab.get('[PAD]') ?? 0
  }

  encode(text: string, maxLength = MAX_SEQ_LEN): { ids: number[]; offsets: [number, number][] } {
    // Basic BERT pre-tokenization: lowercase, handle whitespace + punctuation
    const cleaned = text.toLowerCase().replace(/\s+/g, ' ').trim()
    const ids: number[] = [this.clsId]
    const offsets: [number, number][] = [[-1, -1]]

    let charPos = 0
    const words = cleaned.split(' ')
    for (const word of words) {
      if (!word) { charPos++; continue }
      const wordTokens = this._wordpiece(word)
      for (const token of wordTokens) {
        if (ids.length >= maxLength - 1) break
        ids.push(this.vocab.get(token) ?? this.unkId)
        offsets.push([charPos, charPos + word.length])
      }
      charPos += word.length + 1
    }
    ids.push(this.sepId)
    offsets.push([-1, -1])
    return { ids, offsets }
  }

  private _wordpiece(word: string): string[] {
    if (this.vocab.has(word)) return [word]
    const pieces: string[] = []
    let start = 0
    while (start < word.length) {
      let end = word.length
      let found = false
      const prefix = start === 0 ? '' : '##'
      while (end > start) {
        const substr = prefix + word.slice(start, end)
        if (this.vocab.has(substr)) {
          pieces.push(substr)
          start = end
          found = true
          break
        }
        end--
      }
      if (!found) { pieces.push('[UNK]'); break }
    }
    return pieces.length > 0 ? pieces : ['[UNK]']
  }

  get CLS_ID() { return this.clsId }
  get SEP_ID() { return this.sepId }
  get PAD_ID() { return this.padId }
}

// ─── Text Processing ──────────────────────────────────────────────────────────

function splitSentences(text: string): string[] {
  // Split on sentence boundaries (period/? /! followed by space + capital)
  return text
    .replace(/\r\n/g, '\n')
    .split(/(?<=[.?!])\s+(?=[A-Z])/g)
    .map(s => s.trim())
    .filter(s => s.length > 15)
}

function detectSection(line: string): string | null {
  const lower = line.toLowerCase().trim()
  for (const [alias, canonical] of Object.entries(SECTION_ALIASES)) {
    if (lower === alias || lower.startsWith(alias + ' ') || lower.startsWith(alias + ':')) {
      return canonical
    }
  }
  return null
}

function sectionDocument(text: string): SentenceRecord[] {
  const lines = text.split('\n')
  const records: SentenceRecord[] = []
  let currentSection = 'intro'
  let sentIdx = 0

  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) continue

    // Detect section headers (short lines, often all-caps or title-case)
    const detected = detectSection(trimmed)
    if (detected && trimmed.length < 80) {
      currentSection = detected
      continue
    }

    // Split content into sentences
    const sents = splitSentences(trimmed + ' ')
    for (const sent of sents) {
      if (sent.length < 20) continue
      records.push({ text: sent, section: currentSection, sentence_index: sentIdx++ })
    }
  }
  return records
}

function chunkText(text: string): string[] {
  const chunks: string[] = []
  let start = 0
  const total = text.length
  while (start < total) {
    let end = Math.min(start + CHUNK_SIZE, total)
    if (end < total) {
      const period = text.indexOf('.', end)
      if (period > 0 && period - end < 150) end = period + 1
    }
    const chunk = text.slice(start, end).trim()
    if (chunk.length > 20) chunks.push(chunk)
    const nextStart = end - CHUNK_OVERLAP
    start = nextStart <= start ? start + CHUNK_SIZE : nextStart
  }
  return chunks
}

// ─── Tensor Helpers ────────────────────────────────────────────────────────────

function makeTensor(data: number[], shape: number[], dtype: 'int64' | 'float32' | 'bool'): any {
  if (dtype === 'int64') return new ort.Tensor('int64', BigInt64Array.from(data.map(BigInt)), shape)
  if (dtype === 'float32') return new ort.Tensor('float32', Float32Array.from(data), shape)
  if (dtype === 'bool') return new ort.Tensor('bool', Uint8Array.from(data), shape)
  throw new Error(`Unsupported dtype: ${dtype}`)
}


function ones(n: number): number[] { return Array(n).fill(1) }

// ─── ONNX Inference Sessions ─────────────────────────────────────────────────

let kwSession: any = null
let stSession: any = null
let kwTokenizer: WordPieceTokenizer | null = null
let stTokenizer: WordPieceTokenizer | null = null

async function loadModels(): Promise<void> {
  post('PROGRESS', { stage: 'Loading ONNX models…', percent: 5 })

  // Load vocab files
  const [kwVocab, stVocab] = await Promise.all([
    fetch(MODEL_BASE + 'keyword_vocab.json').then(r => r.json()),
    fetch(MODEL_BASE + 'structure_vocab.json').then(r => r.json()),
  ])
  kwTokenizer = new WordPieceTokenizer(kwVocab)
  stTokenizer = new WordPieceTokenizer(stVocab)

  post('PROGRESS', { stage: 'Loading keyword model…', percent: 15 })

  // Configure ONNX Runtime Web environment
  const numThreads = Math.min(8, self.navigator.hardwareConcurrency || 8)
  ort.env.wasm.numThreads = numThreads
  ort.env.wasm.simd = true
  ort.env.wasm.proxy = false

  // Configure PDF.js worker to dynamically match the installed API version
  pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`

  // Create ONNX Runtime sessions with WebAssembly backend
  const sessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    intraOpNumThreads: numThreads,
    interOpNumThreads: 1
  }

  kwSession = await ort.InferenceSession.create(
    MODEL_BASE + 'keyword_extractor_int8.onnx',
    sessionOptions,
  )

  post('PROGRESS', { stage: 'Loading structure model…', percent: 40 })

  stSession = await ort.InferenceSession.create(
    MODEL_BASE + 'structure_model_int8.onnx',
    sessionOptions,
  )

  post('PROGRESS', { stage: 'Models loaded ✓', percent: 55 })
}

// ─── Keyword Inference ────────────────────────────────────────────────────────

interface BIOPrediction {
  sentence_idx: number
  start_char: number
  end_char: number
  boundary_score: number
  sentence_score: number
  surface: string
}

async function runKeywordInference(sentences: string[]): Promise<BIOPrediction[]> {
  if (!kwSession || !kwTokenizer) throw new Error('Keyword model not loaded')

  const allPredictions: BIOPrediction[] = []
  const windowSize = MAX_SENTENCES_KW
  const totalWindows = Math.ceil(sentences.length / windowSize)
  let currentWindow = 0

  for (let winStart = 0; winStart < sentences.length; winStart += windowSize) {
    currentWindow++
    post('PROGRESS', { 
      stage: `Running keyword extractor… (${currentWindow}/${totalWindows})`, 
      percent: 68 + Math.floor(12 * (currentWindow / totalWindows)) 
    })
    const windowSentences = sentences.slice(winStart, winStart + windowSize)
    if (windowSentences.length === 0) continue

    // Build window token sequence: [CLS] sent1 [SEP] [CLS] sent2 [SEP] ...
    const inputIds: number[] = []
    const attentionMask: number[] = []
    const tokenTypeIds: number[] = []
    const clsPositions: number[] = []
    // Track which (sentence_idx, char_start, char_end) each token came from
    const tokenMeta: Array<{ sent_idx: number; start: number; end: number } | null> = []

    for (let localIdx = 0; localIdx < windowSentences.length; localIdx++) {
      const globalIdx = winStart + localIdx
      const sent = windowSentences[localIdx]
      const { ids, offsets } = kwTokenizer.encode(sent, MAX_SEQ_LEN - 2)

      // Only the non-CLS non-SEP subword tokens (indices 1..ids.length-2)
      const sentIds = ids.slice(1, -1)
      const sentOffsets = offsets.slice(1, -1)

      if (sentIds.length === 0) continue
      if (inputIds.length + sentIds.length + 2 > MAX_SEQ_LEN) break

      const segId = localIdx % 2
      clsPositions.push(inputIds.length)

      // CLS token
      inputIds.push(kwTokenizer.CLS_ID)
      attentionMask.push(1)
      tokenTypeIds.push(segId)
      tokenMeta.push(null)

      for (let t = 0; t < sentIds.length; t++) {
        inputIds.push(sentIds[t])
        attentionMask.push(1)
        tokenTypeIds.push(segId)
        tokenMeta.push({ sent_idx: globalIdx, start: sentOffsets[t][0], end: sentOffsets[t][1] })
      }

      // SEP token
      inputIds.push(kwTokenizer.SEP_ID)
      attentionMask.push(1)
      tokenTypeIds.push(segId)
      tokenMeta.push(null)
    }

    if (inputIds.length === 0 || clsPositions.length === 0) continue

    const numSent = clsPositions.length
    const seqLen = inputIds.length

    const feeds = {
      input_ids:      makeTensor(inputIds, [1, seqLen], 'int64'),
      attention_mask: makeTensor(attentionMask, [1, seqLen], 'int64'),
      token_type_ids: makeTensor(tokenTypeIds, [1, seqLen], 'int64'),
      cls_positions:  makeTensor(clsPositions, [1, numSent], 'int64'),
      sentence_mask:  makeTensor(ones(numSent), [1, numSent], 'bool'),
    }

    const results = await kwSession.run(feeds)
    const sentProbs = Array.from(results.sentence_probs.data as Float32Array)
    const boundaryProbs = Array.from(results.boundary_probs.data as Float32Array)
    // boundary_probs shape: [1, seqLen, 3] — reshape
    const bProbs3D: number[][][] = [Array.from({ length: seqLen }, (_, i) => [
      boundaryProbs[i * 3], boundaryProbs[i * 3 + 1], boundaryProbs[i * 3 + 2],
    ])]

    // BIO decoding: find B→I spans
    let t = 0
    while (t < seqLen) {
      const meta = tokenMeta[t]
      if (!meta) { t++; continue }
      const bProb = bProbs3D[0][t][1] // class 1 = B
      const isB = bProb > 0.5
      if (!isB) { t++; continue }

      let endT = t + 1
      while (
        endT < seqLen &&
        tokenMeta[endT] !== null &&
        (tokenMeta[endT]!).sent_idx === meta.sent_idx &&
        bProbs3D[0][endT][2] > 0.5  // class 2 = I
      ) {
        endT++
      }

      const spanMeta = tokenMeta[endT - 1]
      if (!spanMeta) { t = endT; continue }

      // Score = mean of B/I probs in span
      let boundaryScore = 0
      for (let k = t; k < endT; k++) {
        boundaryScore += Math.max(bProbs3D[0][k][1], bProbs3D[0][k][2])
      }
      boundaryScore /= (endT - t)

      // relative within window (unused)
      const globalSentIdx = meta.sent_idx
      const sentIdx = globalSentIdx < sentProbs.length ? globalSentIdx : 0

      const surface = sentences[globalSentIdx]
        ? sentences[globalSentIdx].slice(meta.start, spanMeta.end).trim()
        : ''

      if (surface.length >= 3) {
        allPredictions.push({
          sentence_idx: globalSentIdx,
          start_char: meta.start,
          end_char: spanMeta.end,
          boundary_score: boundaryScore,
          sentence_score: sentProbs[Math.min(sentIdx, sentProbs.length - 1)] ?? 0,
          surface,
        })
      }

      t = endT
    }
  }

  return allPredictions
}

// ─── Structure Inference ──────────────────────────────────────────────────────

interface SentenceStructure {
  role: string
  role_score: number
  evidence_score: number
  importance: number
}

async function runStructureInference(
  records: SentenceRecord[],
): Promise<Map<number, SentenceStructure>> {
  if (!stSession || !stTokenizer) throw new Error('Structure model not loaded')

  const predictions = new Map<number, SentenceStructure>()
  const windowSize = MAX_SENTENCES_ST
  const totalWindows = Math.ceil(records.length / windowSize)
  let currentWindow = 0

  for (let winStart = 0; winStart < records.length; winStart += windowSize) {
    currentWindow++
    post('PROGRESS', { 
      stage: `Running structure classifier… (${currentWindow}/${totalWindows})`, 
      percent: 80 + Math.floor(10 * (currentWindow / totalWindows)) 
    })
    const windowRecords = records.slice(winStart, winStart + windowSize)

    const inputIds: number[] = []
    const attentionMask: number[] = []
    const tokenTypeIds: number[] = []
    const sectionTokenIds: number[] = []
    const clsPositions: number[] = []
    const globalIndices: number[] = []

    for (let localIdx = 0; localIdx < windowRecords.length; localIdx++) {
      const rec = windowRecords[localIdx]
      const globalIdx = winStart + localIdx
      const { ids } = stTokenizer.encode(rec.text, MAX_SEQ_LEN - 2)
      const sentIds = ids.slice(1, -1)
      if (sentIds.length === 0) continue
      if (inputIds.length + sentIds.length + 2 > MAX_SEQ_LEN) break

      const segId = localIdx % 2
      const sectionId = CANONICAL_SECTIONS.indexOf(rec.section)
      const sectionIdx = sectionId >= 0 ? sectionId : 0

      clsPositions.push(inputIds.length)
      globalIndices.push(globalIdx)

      inputIds.push(stTokenizer.CLS_ID)
      attentionMask.push(1)
      tokenTypeIds.push(segId)
      sectionTokenIds.push(sectionIdx)

      for (const id of sentIds) {
        inputIds.push(id)
        attentionMask.push(1)
        tokenTypeIds.push(segId)
        sectionTokenIds.push(sectionIdx)
      }

      inputIds.push(stTokenizer.SEP_ID)
      attentionMask.push(1)
      tokenTypeIds.push(segId)
      sectionTokenIds.push(sectionIdx)
    }

    if (inputIds.length === 0 || clsPositions.length === 0) continue

    const numSent = clsPositions.length
    const seqLen = inputIds.length

    const feeds = {
      input_ids:         makeTensor(inputIds, [1, seqLen], 'int64'),
      attention_mask:    makeTensor(attentionMask, [1, seqLen], 'int64'),
      token_type_ids:    makeTensor(tokenTypeIds, [1, seqLen], 'int64'),
      section_token_ids: makeTensor(sectionTokenIds, [1, seqLen], 'int64'),
      cls_positions:     makeTensor(clsPositions, [1, numSent], 'int64'),
      sentence_mask:     makeTensor(ones(numSent), [1, numSent], 'bool'),
    }

    const results = await stSession.run(feeds)
    const roleProbs = Array.from(results.role_probs.data as Float32Array)
    const evidenceProbs = Array.from(results.evidence_probs.data as Float32Array)
    const importanceScores = Array.from(results.importance_scores.data as Float32Array)

    const numRoles = ROLE_LABELS.length

    for (let s = 0; s < globalIndices.length; s++) {
      const globalIdx = globalIndices[s]
      const roleSlice = roleProbs.slice(s * numRoles, (s + 1) * numRoles)
      let bestRoleIdx = 0
      let bestRoleScore = -Infinity
      for (let r = 0; r < roleSlice.length; r++) {
        if (roleSlice[r] > bestRoleScore) { bestRoleScore = roleSlice[r]; bestRoleIdx = r }
      }

      predictions.set(globalIdx, {
        role: ROLE_LABELS[bestRoleIdx] ?? 'none',
        role_score: bestRoleScore,
        evidence_score: evidenceProbs[s] ?? 0,
        importance: importanceScores[s] ?? 0,
      })
    }
  }

  return predictions
}

// ─── Concept Unit Builder ─────────────────────────────────────────────────────

function buildConceptUnits(
  sentences: string[],
  records: SentenceRecord[],
  bioPredictions: BIOPrediction[],
  structurePreds: Map<number, SentenceStructure>,
  topK = 28,
): ConceptUnit[] {
  // Deduplicate BIO predictions by surface form
  const surfaceMap = new Map<string, BIOPrediction>()
  for (const pred of bioPredictions) {
    const key = pred.surface.toLowerCase().replace(/\s+/g, ' ')
    const existing = surfaceMap.get(key)
    if (!existing || pred.boundary_score * 0.65 + pred.sentence_score * 0.35 >
        (existing.boundary_score * 0.65 + existing.sentence_score * 0.35)) {
      surfaceMap.set(key, pred)
    }
  }

  // Score and sort
  const scored: Array<BIOPrediction & { combined_score: number }> = []
  for (const pred of surfaceMap.values()) {
    const combined = 0.65 * pred.boundary_score + 0.35 * pred.sentence_score
    scored.push({ ...pred, combined_score: combined })
  }
  scored.sort((a, b) => b.combined_score - a.combined_score)

  // Build concept units using coverage re-ranking (diversity)
  const selected: ConceptUnit[] = []
  const pool = scored.slice(0, topK * 4)

  while (pool.length > 0 && selected.length < topK) {
    let bestIdx = 0
    let bestScore = -Infinity

    for (let i = 0; i < pool.length; i++) {
      const p = pool[i]
      const maxRedundancy = selected.reduce((max, sel) => {
        const jaccard = tokenJaccard(p.surface, sel.phrase)
        return Math.max(max, jaccard)
      }, 0)
      const sentRedundancy = selected.some(sel => sel.sentence_index === p.sentence_idx) ? 0.15 : 0
      const adjustedScore = 0.72 * p.combined_score - 0.28 * maxRedundancy - sentRedundancy
      if (adjustedScore > bestScore) { bestScore = adjustedScore; bestIdx = i }
    }

    const chosen = pool.splice(bestIdx, 1)[0]
    const sp = structurePreds.get(chosen.sentence_idx)
    const rec = records[chosen.sentence_idx]

    selected.push({
      section: rec?.section ?? 'intro',
      phrase: chosen.surface,
      role: sp?.role ?? 'none',
      evidence_sentence: sentences[chosen.sentence_idx] ?? '',
      importance: sp?.importance ?? 0,
      sentence_index: chosen.sentence_idx,
      boundary_score: chosen.boundary_score,
      evidence_score: sp?.evidence_score ?? 0,
      role_score: sp?.role_score ?? 0,
    })
  }

  return selected
}

function tokenJaccard(a: string, b: string): number {
  const setA = new Set(a.toLowerCase().split(/\s+/))
  const setB = new Set(b.toLowerCase().split(/\s+/))
  let intersection = 0
  for (const t of setA) if (setB.has(t)) intersection++
  const union = setA.size + setB.size - intersection
  return union === 0 ? 0 : intersection / union
}

function buildSectionNotes(
  records: SentenceRecord[],
  structurePreds: Map<number, SentenceStructure>,
): Record<string, string[]> {
  const notes: Record<string, string[]> = {}
  const IMPORTANT_ROLES = new Set(['objective', 'core_method', 'result', 'contribution', 'finding'])

  for (const [idx, sp] of structurePreds.entries()) {
    const rec = records[idx]
    if (!rec) continue
    if (!IMPORTANT_ROLES.has(sp.role)) continue
    if (sp.importance < 0.45) continue

    const section = rec.section
    if (!notes[section]) notes[section] = []
    if (notes[section].length < 3) notes[section].push(rec.text)
  }
  return notes
}

// ─── PDF Text Extraction (PDF.js) ─────────────────────────────────────────────
async function extractTextFromPdf(pdfBuffer: ArrayBuffer): Promise<string> {
  const pdf = await pdfjsLib.getDocument({ data: pdfBuffer }).promise
  const textParts: string[] = []

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum)
    const content = await page.getTextContent()
    const pageText = content.items
      .map((item: any) => ('str' in item ? item.str : ''))
      .join(' ')
    textParts.push(pageText)
  }

  return textParts.join('\n')
}

// ─── Main Inference Pipeline ──────────────────────────────────────────────────

async function runInferencePipeline(pdf: ArrayBuffer, filename: string): Promise<EvidencePayload> {
  // Step A: Extract text
  post('PROGRESS', { stage: 'Extracting PDF text…', percent: 58 })
  const fullText = await extractTextFromPdf(pdf)

  if (!fullText.trim()) {
    throw new Error('PDF appears to be empty or image-only (no extractable text).')
  }

  // Step B: Section-aware document parsing
  post('PROGRESS', { stage: 'Parsing document structure…', percent: 62 })
  const records = sectionDocument(fullText)
  const sentences = records.map(r => r.text)
  const sectionsFound = [...new Set(records.map(r => r.section))]

  // Step C: Keyword / BIO inference
  post('PROGRESS', { stage: 'Running keyword extractor…', percent: 68 })
  const bioPredictions = await runKeywordInference(sentences)

  // Step D: Structure inference
  post('PROGRESS', { stage: 'Running structure classifier…', percent: 80 })
  const structurePreds = await runStructureInference(records)

  // Step E: Build concept units
  post('PROGRESS', { stage: 'Building concept units…', percent: 90 })
  const units = buildConceptUnits(sentences, records, bioPredictions, structurePreds, 28)
  const sectionNotes = buildSectionNotes(records, structurePreds)

  // Step F: Build plain text chunks for the backend RAG fallback
  const plainChunks = chunkText(fullText)

  post('PROGRESS', { stage: 'Done ✓', percent: 100 })

  return {
    title: filename.replace(/\.pdf$/i, ''),
    total_sentences: sentences.length,
    sections_found: sectionsFound,
    units,
    section_notes: sectionNotes,
    chunk_text: plainChunks,
  }
}

// ─── Worker Message Bus ────────────────────────────────────────────────────────

function post(type: string, payload: any) {
  self.postMessage({ type, payload })
}

self.addEventListener('message', async (event: MessageEvent<WorkerMessage>) => {
  const { type, payload } = event.data

  try {
    if (type === 'INIT') {
      // Preload ONNX models on app start (warm-up for IndexedDB cache)
      await loadModels()
      post('PROGRESS', { stage: 'Models ready', percent: 100 })
    } else if (type === 'INFER') {
      if (!payload?.pdf) throw new Error('No PDF buffer provided')

      // Load models if not already loaded
      if (!kwSession || !stSession) {
        await loadModels()
      }

      const result = await runInferencePipeline(payload.pdf, payload.filename ?? 'paper.pdf')
      self.postMessage({ type: 'RESULT', payload: result })
    } else {
      throw new Error(`Unknown message type: ${type}`)
    }
  } catch (err: any) {
    post('ERROR', { message: err?.message ?? String(err) })
  }
})
