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
 *    keyword_extractor_int8.onnx or keyword_extractor_int8.onnx.gz
 *    keyword_vocab.json
 *    structure_model_int8.onnx or structure_model_int8.onnx.gz
 *    structure_vocab.json
 *    stage3_resources.json
 *    manifest.json
 *
 * Sectioning system ported from Model/PaperCard/src/ske/structure/sectioning.py
 * (commit b015fa1f6f18e9109ef5e9a6079b11e31443f56f – "Update PaperCard parser and
 *  regenerated evidence outputs").
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
  chunk_text: string[]   // ← plain text chunks for backend RAG fallback
}

interface SentenceRecord {
  text: string
  section: string
  section_title: string
  sentence_index: number
}

interface Stage3BowTerm {
  section: string
  canonical: string
  display: string
  aliases: string[]
  confidence: number
}

interface Stage3TfidfTerm {
  section: string
  term: string
  idf: number
}

interface Stage3Resources {
  schema_version: string
  bow_terms: Stage3BowTerm[]
  tfidf_terms: Stage3TfidfTerm[]
}

interface BowMatch {
  term: Stage3BowTerm
  alias: string
  match_quality: number
  bow_support_score: number
}

interface TfidfMatch {
  section: string
  term: string
  matched_tfidf: number
  max_tfidf: number
  tfidf_support_score: number
}

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

// ─── Constants ────────────────────────────────────────────────────────────────

const MODEL_BASE = '/models/'
const MAX_SEQ_LEN = 512
const MAX_SENTENCES_KW = 16
const MAX_SENTENCES_ST = 48
const CHUNK_SIZE = 800    // chars, mirrors backend chunk_text()
const CHUNK_OVERLAP = 100
const STAGE1_CANDIDATE_WEIGHTS = { boundary: 0.65, selector: 0.25, bow: 0.10 }
const STAGE1_RERANK_WEIGHTS = { candidate: 0.75, coverage: 0.25 }
const CONCEPT_IMPORTANCE_WEIGHTS = { rerank: 0.50, evidence: 0.25, sentence: 0.25 }
const STAGE3_THRESHOLDS = {
  phrase_word_number: 2,
  bow_support_score: 0.70,
  tfidf_support_score: 0.50,
}
const FINAL_TOP_K = 28
const CANDIDATE_POOL_MULTIPLIER = 4

const CANONICAL_SECTIONS = ['intro', 'related_work', 'method', 'experiment', 'conclusion']
const ROLE_LABELS = [
  'none', 'background', 'problem', 'motivation', 'objective',
  'prior_work', 'limitation', 'gap', 'comparison',
  'core_method', 'component', 'mechanism', 'process',
  'dataset', 'metric', 'baseline', 'result', 'ablation',
  'contribution', 'finding', 'future_work',
]

// ─── Sectioning System (ported from sectioning.py) ────────────────────────────
//
// The v4 structure model (structure_v4_partial_role_balanced_fulldev) was trained
// with Candidate Label Loss – it is extremely sensitive to the section_token_ids
// passed as prior context. The old SECTION_ALIASES dict produced too many
// "intro" assignments, causing category chaos. The new system ported from Python
// correctly classifies headings via multi-pattern fullmatch regex with length/
// word-count guards, noise-line filtering, terminal-section early exit, inline
// section detection for single-block PDFs, and positional fall-back repair.

/** Matches isolated section number lines like "1", "2.1", "A." */
const SECTION_NUMBER_RE = /^\s*(?:\d+(?:\.\d+)*|[A-Z])\.?\s*$/

/** Matches bare page number lines (including form-feed chars normalised away) */
const PAGE_NUMBER_RE = /^\s*\d{1,3}\s*$/

/** Strips optional leading number and captures the rest of a heading line */
const SECTION_HEADING_RE = /^\s*(?:\d+(?:\.\d+)*\.?\s+)?(.+?)\s*$/i

/** Headings that terminate the body of the paper (references etc.) */
const TERMINAL_HEADING_RE =
  /^\s*(?:\d+(?:\.\d+)*\.?\s+)?(references|bibliography|acknowledg(?:e)?ments?|appendix|appendices|supplementary(?: material| information)?|attention visualizations?)\b/i

/** Numbered inline headings embedded inside running text (no newline separators) */
const NUMBERED_INLINE_HEADING_RE =
  /(?<![a-z0-9])(?<number>[1-9](?:\.\d{1,2}){0,2})\s+(?<title>(?:[a-z][a-z0-9+\-]*\s+){0,8}?(?:related work|background|preliminaries|methodology|methods?|approach|framework|architecture|architectures|models?|algorithms?|metrics?|evaluation|evaluations|experiments?|results?|analysis|discussion|datasets?|benchmarks?|training|examples?|conclusions?|future work|references|acknowledg(?:e)?ments?|appendix|appendices))\b/gi

/** Regex for conclusion tail heuristic */
const CONCLUSION_TAIL_RE =
  /\b(in this work|in this paper|we (?:presented|introduced|demonstrated|showed|have shown|conclude)|future work)\b/i

/**
 * HEADING_SECTION_PATTERNS mirrors the Python tuple-of-tuples structure:
 *   [section_id, display_title, [...regex_patterns_for_fullmatch]]
 *
 * Patterns are applied with re.fullmatch semantics (anchored at both ends)
 * after normalize_heading_text() has been applied.
 */
const HEADING_SECTION_PATTERNS: Array<[string, string, RegExp[]]> = [
  ['intro', 'Abstract', [
    /abstract/, /introduction/, /intro/, /overview/,
  ]],
  ['related_work', 'Related Work', [
    /background/, /related work/, /prior work/, /literature review/, /preliminaries/,
  ]],
  ['method', 'Method', [
    /method/, /methods/, /methodology/, /approach/, /proposed method/,
    /model/, /model architecture/, /architecture/, /framework/, /algorithm/,
    /update rule/, /initialization bias correction/, /convergence analysis/,
    /theoretical analysis/, /proof/, /derivation/, /encoder and decoder stacks/,
    /encoder/, /decoder/, /attention/, /self-attention/, /multi-head attention/,
    /scaled dot-product attention/, /applications of attention in our model/,
    /position-wise feed-forward networks/, /positional encoding/, /objective/,
    /loss/, /why self-attention/,
  ]],
  ['experiment', 'Experiment', [
    /experiments?/, /experimental setup/, /empirical results?/, /evaluation/,
    /results?/, /discussion/, /ablation(?: studies)?/, /analysis/, /training/,
    /training data and batching/, /hardware and schedule/, /optimizer/,
    /regularization/, /model variations/, /machine translation/,
    /english constituency parsing/, /logistic regression/,
    /multi-layer neural networks/, /convolutional neural networks/,
    /bias-correction term/, /datasets?/, /benchmarks?/,
  ]],
  ['conclusion', 'Conclusion', [
    /conclusion/, /conclusions/, /limitations?/, /future work/,
  ]],
]

// ─── Inline-marker system (ported from find_inline_markers + helpers) ─────────

interface InlineMarker {
  start: number
  end: number
  title: string
  section: string
  priority: number
  terminal: boolean
}

// ─── PDF noise filtering ──────────────────────────────────────────────────────

function cleanPdfLines(text: string): string[] {
  const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').replace(/\f/g, '\n')
  return normalized.split('\n').map(normalizePdfLine)
}

function normalizePdfLine(line: string): string {
  return line.replace(/\s+/g, ' ').trim().replace(/^\uFEFF/, '')
}

/**
 * Returns true for lines that should be ignored during heading detection:
 *  – empty lines
 *  – standalone section/page numbers ("1", "2.3", "42")
 *  – conference paper attribution lines ("Published as a conference paper at …")
 *  – arXiv header lines ("arXiv:2305.xxxxx")
 */
function isNoiseLine(line: string): boolean {
  if (!line) return true
  if (PAGE_NUMBER_RE.test(line) || SECTION_NUMBER_RE.test(line)) return true
  const lower = line.toLowerCase()
  if (lower.startsWith('published as a conference paper at') && line.split(/\s+/).length <= 8) return true
  if (lower.startsWith('arxiv:') && line.split(/\s+/).length <= 5) return true
  return false
}

function isTerminalHeadingLine(line: string): boolean {
  return TERMINAL_HEADING_RE.test(line)
}

/**
 * normalize_heading_text() – mirrors the Python version exactly:
 *  1. Lowercase
 *  2. Strip leading section number (e.g. "3.2 " → "")
 *  3. Strip trailing colon/period
 *  4. Replace non-alphanumeric chars (except +, -, space) with space
 *  5. Collapse whitespace
 */
function normalizeHeadingText(heading: string): string {
  let h = heading.toLowerCase()
  h = h.replace(/^\s*(?:\d+(?:\.\d+)*|[a-z])\.?\s+/, '')
  h = h.replace(/[:.]\s*$/, '')
  h = h.replace(/[^a-z0-9+\-\s]/g, ' ')
  h = h.replace(/\s+/g, ' ').trim()
  return h
}

/**
 * detect_section_heading() – returns [display_title, section_id] or null.
 *
 * Guards:
 *  – line must be ≤9 words (section headings are short)
 *  – applies SECTION_HEADING_RE to strip leading numbers
 *  – normalizes with normalizeHeadingText()
 *  – fullmatches against each HEADING_SECTION_PATTERNS entry
 */
function detectSectionHeading(line: string): [string, string] | null {
  if (line.split(/\s+/).length > 9) return null
  const match = SECTION_HEADING_RE.exec(line)
  if (!match) return null
  const headingText = normalizeHeadingText(match[1])
  if (!headingText) return null
  for (const [section, , patterns] of HEADING_SECTION_PATTERNS) {
    for (const pat of patterns) {
      // Simulate fullmatch: anchor pattern
      const fullPat = new RegExp(`^(?:${pat.source})$`, 'i')
      if (fullPat.test(headingText)) {
        return [line, section]
      }
    }
  }
  return null
}

/**
 * normalize_heading_to_section() – coarse token-based fallback used by
 * the numbered inline heading system.
 */
function normalizeHeadingToSection(heading: string): string {
  const h = normalizeHeadingText(heading)
  if (/\b(related|prior|literature|background|preliminaries)\b/.test(h)) return 'related_work'
  if (/\b(conclusion|limitation|future)\b/.test(h)) return 'conclusion'
  if (/\b(method|approach|model|architecture|algorithm|metric|metrics|pre-training|fine-tuning|pretraining|finetuning|masked lm|next sentence prediction|input output representations|convergence|theoretical|proof|derivation|encoder|decoder|attention|objective|loss)\b/.test(h)) return 'method'
  if (/\b(experiment|evaluation|human evaluation|bleu evaluation|result|discussion|training|regularization|optimizer|dataset|benchmark|ablation|comparison|examples)\b/.test(h)) return 'experiment'
  return 'intro'
}

// ─── Section document (main line-based parser) ────────────────────────────────

interface SectionChunk {
  section: string
  title: string
  text: string
}

function splitIntoSectionChunks(text: string): SectionChunk[] {
  const lines = cleanPdfLines(text)
  const chunks: SectionChunk[] = []
  let currentTitle = 'Introduction'
  let currentSection = 'intro'
  const buffer: string[] = []
  let seenHeading = false

  for (const line of lines) {
    if (!line || isNoiseLine(line)) continue

    // Stop at terminal headings (references, acknowledgements, etc.)
    if (isTerminalHeadingLine(line)) {
      if (seenHeading) {
        if (buffer.length > 0) {
          chunks.push({ section: currentSection, title: currentTitle, text: buffer.join(' ') })
        }
        return mergeAdjacentChunks(chunks)
      }
      break
    }

    const heading = detectSectionHeading(line)
    if (heading) {
      seenHeading = true
      if (buffer.length > 0) {
        chunks.push({ section: currentSection, title: currentTitle, text: buffer.join(' ') })
        buffer.length = 0
      }
      ;[currentTitle, currentSection] = heading
      continue
    }
    buffer.push(line)
  }

  if (buffer.length > 0) {
    chunks.push({ section: currentSection, title: currentTitle, text: buffer.join(' ') })
  }

  if (seenHeading) return chunks

  // Fall back to inline marker detection for single-block PDFs
  const inlineChunks = splitInlineChunks(text)
  if (inlineChunks.length > 0) return inlineChunks

  return [{ section: 'intro', title: 'Unsectioned', text }]
}

function mergeAdjacentChunks(chunks: SectionChunk[]): SectionChunk[] {
  const merged: SectionChunk[] = []
  for (const chunk of chunks) {
    if (merged.length > 0 && merged[merged.length - 1].section === chunk.section && merged[merged.length - 1].title === chunk.title) {
      merged[merged.length - 1].text = `${merged[merged.length - 1].text} ${chunk.text}`.trim()
    } else {
      merged.push({ ...chunk })
    }
  }
  return merged
}

// ─── Positional repair passes ─────────────────────────────────────────────────

/**
 * assign_positional_sections() – If the document has no recognisable sections
 * at all (everything fell back to "intro/Unsectioned"), slice it proportionally.
 * Mirrors the Python implementation exactly.
 */
function assignPositionalSections(sentences: SentenceRecord[]): SentenceRecord[] {
  const hasRealSections = sentences.some(
    s => s.section !== 'intro' || s.section_title !== 'Unsectioned',
  )
  if (hasRealSections || sentences.length < 8) return sentences

  const total = sentences.length
  const boundaries: Array<[string, number]> = [
    ['intro', 0.20],
    ['related_work', 0.35],
    ['method', 0.65],
    ['experiment', 0.90],
    ['conclusion', 1.01],
  ]
  for (let idx = 0; idx < sentences.length; idx++) {
    const ratio = idx / Math.max(total, 1)
    for (const [section, endRatio] of boundaries) {
      if (ratio < endRatio) {
        sentences[idx].section = section
        sentences[idx].section_title = 'Positional ' + section
        break
      }
    }
  }
  return sentences
}

/**
 * repair_related_work_bridge() – If no related_work section was found but there
 * is a method section, scan the sentences just before it and reclassify any that
 * look like "prior work" discourse as related_work.
 */
function repairRelatedWorkBridge(sentences: SentenceRecord[]): SentenceRecord[] {
  const hasRelated = sentences.some(s => s.section === 'related_work')
  if (hasRelated || sentences.length < 10) return sentences
  const firstMethod = sentences.findIndex(s => s.section === 'method')
  if (firstMethod === null || firstMethod < 4) return sentences
  const start = Math.max(2, firstMethod - 5)
  for (let i = start; i < firstMethod; i++) {
    if (looksLikeRelatedWorkBridge(sentences[i].text)) {
      sentences[i].section = 'related_work'
      sentences[i].section_title = 'Inferred related-work bridge'
    }
  }
  return sentences
}

function looksLikeRelatedWorkBridge(text: string): boolean {
  const lower = text.toLowerCase()
  const bridgeCues = ['previous', 'prior', 'existing', 'recent', 'earlier', 'before ', 'although', 'however', 'compared with', 'not fine-tuned', 'additional pretraining', 'tf-idf', 'bm25', 'orqa', 'baseline']
  if (!bridgeCues.some(cue => lower.includes(cue))) return false
  if (['we propose', 'we introduce', 'we present', 'our method', 'our model'].some(cue => lower.includes(cue))) return false
  return true
}

/**
 * repair_conclusion_tail() – If no conclusion was detected but the last ~28% of
 * the doc has sentences matching conclusory phrases, reclassify them.
 */
function repairConclusionTail(sentences: SentenceRecord[]): SentenceRecord[] {
  const hasConclusion = sentences.some(s => s.section === 'conclusion')
  if (hasConclusion || sentences.length < 10) return sentences
  const startIdx = Math.max(0, Math.floor(sentences.length * 0.72))
  for (let i = startIdx; i < sentences.length; i++) {
    const s = sentences[i]
    if (!['method', 'experiment'].includes(s.section)) continue
    if (!CONCLUSION_TAIL_RE.test(s.text)) continue
    for (let j = s.sentence_index; j < sentences.length; j++) {
      if (['method', 'experiment'].includes(sentences[j].section)) {
        sentences[j].section = 'conclusion'
        sentences[j].section_title = 'Inferred conclusion tail'
      }
    }
    break
  }
  return sentences
}

/**
 * repair_low_coverage_sections() – If the parse produced fewer than 3 effective
 * sections (or lacks both method and experiment), fall back to proportional
 * position-based assignment for the body.
 *
 * This is the key safety net introduced in b015fa1 to handle PDFs where the
 * heading detector finds some headings but still misses major sections.
 */
function repairLowCoverageSections(sentences: SentenceRecord[]): SentenceRecord[] {
  if (sentences.length < 40) return sentences

  const counts: Record<string, number> = {}
  for (const s of sentences) counts[s.section] = (counts[s.section] ?? 0) + 1

  const minCount = Math.max(2, Math.floor(sentences.length * 0.03))
  const effectiveSections = CANONICAL_SECTIONS.filter(sec => (counts[sec] ?? 0) >= minCount)
  if (effectiveSections.length > 2 && counts['method'] && counts['experiment']) return sentences

  const conclusionStart = sentences.find(s => s.section === 'conclusion')?.sentence_index ?? null
  const bodyEnd = (conclusionStart !== null && conclusionStart > 8) ? conclusionStart : sentences.length
  if (bodyEnd < 30) return sentences

  const bodyBoundaries: Array<[string, number]> = [
    ['intro', 0.20],
    ['related_work', 0.35],
    ['method', 0.65],
    ['experiment', 1.01],
  ]
  for (let idx = 0; idx < bodyEnd; idx++) {
    const ratio = idx / Math.max(bodyEnd, 1)
    for (const [section, endRatio] of bodyBoundaries) {
      if (ratio < endRatio) {
        sentences[idx].section = section
        sentences[idx].section_title = 'Low-coverage positional ' + section
        break
      }
    }
  }

  if (conclusionStart === null) {
    const tailStart = Math.max(0, Math.floor(sentences.length * 0.92))
    for (let i = tailStart; i < sentences.length; i++) {
      if (CONCLUSION_TAIL_RE.test(sentences[i].text)) {
        for (let j = sentences[i].sentence_index; j < sentences.length; j++) {
          sentences[j].section = 'conclusion'
          sentences[j].section_title = 'Low-coverage inferred conclusion'
        }
        break
      }
    }
  }
  return sentences
}

// ─── Inline section detection (for single-block PDFs) ─────────────────────────
// Ported from split_inline_section_chunks / find_inline_markers / helpers.

function contextWords(text: string): string[] {
  return text.match(/[a-z0-9+\-]+/g) ?? []
}

function isInlineHeadingContext(
  lowered: string, start: number, end: number,
  title: string, terminal: boolean,
): boolean {
  const leftWords = contextWords(lowered.slice(Math.max(0, start - 180), start))
  const rightWords = contextWords(lowered.slice(end, Math.min(lowered.length, end + 220)))
  const previous = leftWords.length ? leftWords[leftWords.length - 1] : ''
  const rightText = rightWords.slice(0, 18).join(' ')
  const phrase = lowered.slice(start, end)

  if (terminal) {
    if (phrase.startsWith('a ') || phrase.startsWith('b ') || phrase.startsWith('c ') || phrase.startsWith('next sentence prediction')) {
      return lowered.slice(Math.max(0, start - 1800), start).includes('conclusion')
    }
    return true
  }

  if (title === 'Abstract') return start < 600 || previous === 'title'
  if (title === 'Introduction') {
    const EMBEDDED = new Set(['a','an','the','this','that','these','those','of','for','in','from','with','using','used','called'])
    return start < Math.max(1800, lowered.length / 4) && !EMBEDDED.has(previous)
  }
  if (title === 'Related Work') return !['in','of','for','from','with'].includes(previous)
  if (title === 'Background') {
    if (['exhaustive','additional','general','model'].includes(previous)) return false
    if (['description','knowledge','information'].includes(rightWords[0])) return false
    return ['goal','foundation','prior','previous','related','history','review'].some(cue => rightText.includes(cue))
  }
  if (title === 'Method') {
    if (['a','an','the','this','that','our','their','same','overall','see'].includes(previous)) return false
    if (phrase === 'approach' && ['based','feature','fine-tuning','finetuning'].includes(previous)) return false
    if (['method','methods'].includes(phrase) && ['from','for','of'].includes(rightWords[0])) return false
    if (['method','methods'].includes(phrase) && previous && !/^\d+$/.test(previous)) return false
    if (['algorithm','algorithms'].includes(phrase) && !['1','2','3','4','5','6','7'].includes(previous)) return false
    const NUMBERED_HEADINGS = new Set([
      'model architecture','adam s update rule','update rule','initialization bias correction',
      'convergence analysis','theoretical analysis','proof','derivation',
      'encoder and decoder stacks','scaled dot-product attention','multi-head attention',
      'applications of attention in our model','position-wise feed-forward networks',
      'positional encoding','why self-attention',
    ])
    if (NUMBERED_HEADINGS.has(phrase) && previous && !/^\d+$/.test(previous)) return false
    const methodCues = ['model','architecture','framework','encoder','decoder','layer','we','this section','competitive','objective','gradient','update','theorem','proof','convergence','attention','input','output','algorithm','pseudo','parameter','parameters','moment']
    return methodCues.some(cue => rightText.includes(cue))
  }
  if (title === 'Experiment') {
    if (['result','results'].includes(phrase) && (['these','our','their','identical','published'].includes(previous) || rightWords[0] === 'see')) return false
    if (['evaluation','analysis'].includes(phrase) && previous && !/^\d+$/.test(previous)) return false
    if (phrase === 'training' && !rightText.startsWith('this section describes')) return false
    if (['experiment','experiments'].includes(phrase) && previous && !/^\d+$/.test(previous)) return false
    if (['optimizer','regularization'].includes(phrase) && previous && !/^\d+$/.test(previous)) return false
    const NUMBERED_HEADINGS = new Set([
      'training data and batching','hardware and schedule','machine translation',
      'english constituency parsing','logistic regression','multi-layer neural networks',
      'convolutional neural networks','bias-correction term','model variations',
    ])
    if (NUMBERED_HEADINGS.has(phrase) && previous && !/^\d+$/.test(previous)) return false
    return true
  }
  if (title === 'Conclusion') return !['in','for','future','our','the'].includes(previous)
  return true
}

function shouldStopAtTerminal(marker: InlineMarker, lowered: string, previous: InlineMarker[]): boolean {
  const ratio = marker.start / Math.max(lowered.length, 1)
  const phrase = lowered.slice(marker.start, marker.end)
  const sawConclusion = previous.some(m => m.section === 'conclusion' && !m.terminal)
  if (sawConclusion) return true
  if (phrase.startsWith('a distant supervision') || phrase.startsWith('b alternative similarity') || phrase.startsWith('c joint training')) return false
  if (phrase.includes('system card')) return ratio > 0.25
  if (phrase.startsWith('references')) return ratio > 0.82
  if (phrase.startsWith('acknowledg')) return ratio > 0.75
  if (phrase.includes('attention visualization')) return ratio > 0.65
  return ratio > 0.75
}

function findInlineMarkers(lowered: string): InlineMarker[] {
  type Spec = [string, string, number, boolean, RegExp[]]
  const specs: Spec[] = [
    ['Abstract', 'intro', 100, false, [/\babstract\b/g]],
    ['Introduction', 'intro', 98, false, [/\bintroduction\b/g]],
    ['Related Work', 'related_work', 95, false, [/\brelated work\b/g, /\bprior work\b/g, /\bliterature review\b/g]],
    ['Background', 'related_work', 84, false, [/\bbackground\b/g]],
    ['Method', 'method', 84, false, [
      /\bmodel architecture\b/g, /\bmethodology\b/g, /\bmethods?\b/g, /\bproposed method\b/g,
      /\balgorithms?\b/g, /\badam s update rule\b/g, /\bupdate rule\b/g,
      /\binitialization bias correction\b/g, /\bconvergence analysis\b/g, /\btheoretical analysis\b/g,
      /\bproof\b/g, /\bderivation\b/g, /\bencoder and decoder stacks\b/g,
      /\bscaled dot-product attention\b/g, /\bmulti-head attention\b/g,
      /\bapplications of attention in our model\b/g, /\bposition-wise feed-forward networks\b/g,
      /\bpositional encoding\b/g, /\bwhy self-attention\b/g,
    ]],
    ['Experiment', 'experiment', 78, false, [
      /\bexperiments?\b/g, /\bexperimental setup\b/g, /\bempirical results?\b/g,
      /\bevaluation\b/g, /\bablation studies\b/g, /\bmodel variations\b/g,
      /\btraining data and batching\b/g, /\bhardware and schedule\b/g, /\boptimizer\b/g,
      /\bregularization\b/g, /\bmachine translation\b/g, /\benglish constituency parsing\b/g,
      /\blogistic regression\b/g, /\bmulti-layer neural networks\b/g,
      /\bconvolutional neural networks\b/g, /\bbias-correction term\b/g,
      /(?<!pre-)\btraining\b/g,
    ]],
    ['Conclusion', 'conclusion', 90, false, [/\bconclusions?\b/g]],
    ['Terminal', 'conclusion', 110, true, [
      /\breferences\b/g, /\backnowledg(?:e)?ments?\b/g, /\bappendix\b/g,
      /\bsupplementary material\b/g, /\bsystem card\b/g, /\battention visualizations?\b/g,
      /\bfull rbrm instructions\b/g, /\ba\s+distant supervision\b/g,
      /\bb\s+alternative similarity\b/g, /\bc\s+joint training\b/g,
      /\bto converge in section [a-z] \d\b/g,
      /\bnext sentence prediction the next sentence prediction task can be illustrated\b/g,
      /\ba\s+\d+\s+(?:pre-training|fine-tuning|comparison|illustrations)\b/g,
      /\bb\s+\d+\s+detailed\b/g, /\bc\s+\d+\s+(?:additional|effect)\b/g,
    ]],
  ]

  const candidates: InlineMarker[] = []
  for (const [title, section, priority, terminal, patterns] of specs) {
    for (const pat of patterns) {
      pat.lastIndex = 0
      let m: RegExpExecArray | null
      while ((m = pat.exec(lowered)) !== null) {
        if (isInlineHeadingContext(lowered, m.index, m.index + m[0].length, title, terminal)) {
          candidates.push({ start: m.index, end: m.index + m[0].length, title, section, priority, terminal })
        }
      }
    }
  }

  // Numbered inline headings
  NUMBERED_INLINE_HEADING_RE.lastIndex = 0
  let nm: RegExpExecArray | null
  while ((nm = NUMBERED_INLINE_HEADING_RE.exec(lowered)) !== null) {
    const rawTitle = nm.groups?.title?.trim() ?? ''
    if (!isNumberedInlineHeadingContext(lowered, nm.index, nm.index + nm[0].length, rawTitle)) continue
    const headingText = normalizeHeadingText(rawTitle)
    if (!headingText) continue
    if (/\b(references|acknowledg|appendix|appendices)\b/.test(headingText)) {
      candidates.push({ start: nm.index, end: nm.index + nm[0].length, title: 'Terminal', section: 'conclusion', priority: 112, terminal: true })
      continue
    }
    const section = normalizeHeadingToSection(headingText)
    const PRIORITY_MAP: Record<string, number> = { intro: 82, related_work: 88, method: 88, experiment: 88, conclusion: 96 }
    const title = headingText.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
    candidates.push({ start: nm.index, end: nm.index + nm[0].length, title, section, priority: PRIORITY_MAP[section] ?? 82, terminal: false })
  }

  // Method transition markers
  candidates.push(...findMethodTransitionMarkers(lowered))

  if (candidates.length === 0) return []

  const sorted = candidates.sort((a, b) => a.start - b.start || b.priority - a.priority)
  const filtered = removePreSectionFalseMarkers(sorted)
  const pruned = pruneInlineMarkers(filtered)
  return coerceInlineSections(pruned, lowered)
}

function isNumberedInlineHeadingContext(lowered: string, start: number, end: number, rawTitle: string): boolean {
  const leftWords = contextWords(lowered.slice(Math.max(0, start - 120), start))
  const rightWords = contextWords(lowered.slice(end, Math.min(lowered.length, end + 120)))
  const previous = leftWords.length ? leftWords[leftWords.length - 1] : ''
  const PREV_STOPWORDS = new Set(['section','sections','figure','fig','table','eq','equation','example','appendix','see','row','rows'])
  if (PREV_STOPWORDS.has(previous)) return false
  const headingText = normalizeHeadingText(rawTitle)
  const firstWord = headingText.split(' ')[0] ?? ''
  if (['so','we','this','these','that','it','because','then'].includes(firstWord)) return false
  const PREP_BLOCKERS = new Set(['of','for','from','with','using','than','about','over','under','between','at'])
  if (PREP_BLOCKERS.has(previous)) {
    const WEAK = new Set(['training','results','experiments','evaluation','analysis','discussion','examples'])
    if (headingText.split(' ').length <= 1 || WEAK.has(headingText)) return false
  }
  if (headingText === 'training' && !rightWords.slice(0, 12).some(w => ['data','regime','procedure','objective','we','the'].includes(w))) return false
  if (['results','discussion','analysis'].includes(headingText) && !/^[1-9]$/.test(previous)) return false
  return true
}

function findMethodTransitionMarkers(lowered: string): InlineMarker[] {
  const markers: InlineMarker[] = []
  const repeated = /\b([a-z][a-z0-9\-]{1,30})\s+we\s+(?:introduce|present|describe)\s+\1\b/g
  let m: RegExpExecArray | null
  while ((m = repeated.exec(lowered)) !== null) {
    if (!['section','paper','model','method','approach'].includes(m[1])) {
      markers.push({ start: m.index, end: m.index + m[1].length, title: 'Method', section: 'method', priority: 76, terminal: false })
    }
  }
  const framework = /\bwe\s+(?:introduce|present|describe)\s+[^.]{0,120}?\bin this section\b/g
  while ((m = framework.exec(lowered)) !== null) {
    markers.push({ start: m.index, end: m.index, title: 'Method', section: 'method', priority: 72, terminal: false })
  }
  const solution = /\b(?:our|the)\s+(?:final\s+)?(?:solution|approach|method|model|framework|retriever)\s+(?:is|uses|consists|relies|optimizes|encodes)\b/g
  while ((m = solution.exec(lowered)) !== null) {
    markers.push({ start: m.index, end: m.index, title: 'Method', section: 'method', priority: 76, terminal: false })
  }
  const implementation = /\bwe\s+(?:use|adopt|train|fine-tune|finetune|optimize|encode)\s+[^.]{0,140}?\b(?:encoder|decoder|retriever|embedding|architecture|objective|loss)\b/g
  while ((m = implementation.exec(lowered)) !== null) {
    markers.push({ start: m.index, end: m.index, title: 'Method', section: 'method', priority: 70, terminal: false })
  }
  return markers
}

function removePreSectionFalseMarkers(candidates: InlineMarker[]): InlineMarker[] {
  const firstIntro = Math.min(...candidates.filter(c => c.title === 'Introduction').map(c => c.start), Infinity)
  const firstRelated = Math.min(...candidates.filter(c => ['Related Work','Background'].includes(c.title)).map(c => c.start), Infinity)
  const firstMethod = Math.min(...candidates.filter(c => c.section === 'method').map(c => c.start), Infinity)
  const firstExperiment = Math.min(...candidates.filter(c => c.section === 'experiment').map(c => c.start), Infinity)
  return candidates.filter(c => {
    if (isFinite(firstIntro) && c.start < firstIntro && c.section !== 'intro' && !c.terminal) return false
    if (isFinite(firstRelated) && c.start < firstRelated && c.section === 'method' && c.priority < 82) return false
    if (isFinite(firstMethod) && c.start < firstMethod && ['experiment','conclusion'].includes(c.section)) return false
    if (isFinite(firstExperiment) && c.start < firstExperiment && c.section === 'conclusion') return false
    return true
  })
}

function pruneInlineMarkers(candidates: InlineMarker[]): InlineMarker[] {
  const kept: InlineMarker[] = []
  for (const candidate of candidates) {
    if (kept.length > 0 && candidate.start - kept[kept.length - 1].start < 80) {
      if (candidate.priority > kept[kept.length - 1].priority) kept[kept.length - 1] = candidate
      continue
    }
    kept.push(candidate)
  }
  return removeSectionRegressions(kept)
}

function removeSectionRegressions(markers: InlineMarker[]): InlineMarker[] {
  const ORDER: Record<string, number> = { intro: 0, related_work: 1, method: 2, experiment: 3, conclusion: 4 }
  const kept: InlineMarker[] = []
  let maxSeen = -1
  for (const marker of markers) {
    if (marker.terminal) { kept.push(marker); continue }
    const sectionOrder = ORDER[marker.section] ?? maxSeen
    if (maxSeen >= ORDER['conclusion'] && marker.section !== 'conclusion') continue
    if (marker.section === 'intro' && maxSeen > ORDER['intro']) continue
    if (sectionOrder < maxSeen && marker.section !== 'related_work') continue
    maxSeen = Math.max(maxSeen, sectionOrder)
    kept.push(marker)
  }
  return kept
}

function coerceInlineSections(markers: InlineMarker[], lowered: string): InlineMarker[] {
  const coerced: InlineMarker[] = []
  let seenIntro = false
  let seenMethod = false
  for (const marker of markers) {
    let { section, title } = marker
    if (title === 'Background') section = seenIntro ? 'related_work' : 'intro'
    if (title === 'Experiment' && lowered.slice(marker.start, marker.end) === 'training' && !seenMethod) section = 'method'
    if (section === 'intro') seenIntro = true
    if (section === 'method') seenMethod = true
    coerced.push({ ...marker, section })
  }
  return coerced
}

function keepEffectiveMarkers(markers: InlineMarker[], compacted: string): InlineMarker[] {
  const effective: InlineMarker[] = []
  for (const marker of markers) {
    if (marker.terminal) {
      if (shouldStopAtTerminal(marker, compacted, effective)) {
        effective.push(marker)
        break
      }
      continue
    }
    effective.push(marker)
  }
  return effective
}

function splitInlineChunks(text: string): SectionChunk[] {
  const compacted = text.replace(/\s+/g, ' ').trim()
  const lowered = compacted.toLowerCase()
  if (lowered.split(/\s+/).length < 220) return []
  const rawMarkers = findInlineMarkers(lowered)
  if (rawMarkers.length < 2) return []
  const markers = keepEffectiveMarkers(rawMarkers, compacted)
  const chunks: SectionChunk[] = []
  if (markers[0].start > 40) {
    chunks.push({ section: 'intro', title: 'Lead', text: compacted.slice(0, markers[0].start).trim() })
  }
  for (let idx = 0; idx < markers.length; idx++) {
    const marker = markers[idx]
    const nextStart = idx + 1 < markers.length ? markers[idx + 1].start : compacted.length
    if (marker.terminal) break
    const chunkText = compacted.slice(marker.end, nextStart).trim()
    if (chunkText.split(/\s+/).length >= 8) {
      chunks.push({ section: marker.section, title: marker.title, text: chunkText })
    }
  }
  return mergeAdjacentChunks(chunks)
}

// ─── Main sectionDocument function ───────────────────────────────────────────

function splitSentences(text: string): string[] {
  return text
    .replace(/\r\n/g, '\n')
    .split(/(?<=[.?!])\s+(?=[A-Z])/g)
    .map(s => s.trim())
    .filter(s => s.length > 15)
}

function sectionDocument(text: string): SentenceRecord[] {
  const chunks = splitIntoSectionChunks(text)
  const sentences: SentenceRecord[] = []

  for (const chunk of chunks) {
    for (const sent of splitSentences(chunk.text)) {
      if (!sent.trim()) continue
      sentences.push({
        text: sent.trim(),
        section: chunk.section,
        section_title: chunk.title,
        sentence_index: sentences.length,
      })
    }
  }

  if (sentences.length === 0) {
    for (const sent of splitSentences(text)) {
      sentences.push({ text: sent, section: 'intro', section_title: 'Unsectioned', sentence_index: sentences.length })
    }
  }

  let repaired = assignPositionalSections(sentences)
  repaired = repairRelatedWorkBridge(repaired)
  repaired = repairConclusionTail(repaired)
  repaired = repairLowCoverageSections(repaired)
  return repaired
}

// ─── Tokenizer (WordPiece, minimal JS implementation) ─────────────────────────

class WordPieceTokenizer {
  private vocab: Map<string, number>
  private unkId: number
  private clsId: number
  private sepId: number
  private padId: number

  constructor(vocab: Record<string, number>) {
    this.vocab = new Map(Object.entries(vocab).map(([k, v]) => [k, v as number]))
    this.unkId = this.vocab.get('[UNK]') ?? 100
    this.clsId = this.vocab.get('[CLS]') ?? 101
    this.sepId = this.vocab.get('[SEP]') ?? 102
    this.padId = this.vocab.get('[PAD]') ?? 0
  }

  encode(text: string, maxLength = MAX_SEQ_LEN): { ids: number[]; offsets: [number, number][] } {
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
let stage3Resources: Stage3Resources = { schema_version: 'empty', bow_terms: [], tfidf_terms: [] }

async function loadStage3Resources(): Promise<Stage3Resources> {
  try {
    const response = await fetch(MODEL_BASE + 'stage3_resources.json')
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return await response.json() as Stage3Resources
  } catch {
    return { schema_version: 'empty', bow_terms: [], tfidf_terms: [] }
  }
}

async function loadOnnxSession(baseName: string, sessionOptions: any): Promise<any> {
  try {
    return await ort.InferenceSession.create(MODEL_BASE + `${baseName}.onnx`, sessionOptions)
  } catch {
    const response = await fetch(MODEL_BASE + `${baseName}.onnx.gz`)
    if (!response.ok) throw new Error(`Unable to load ${baseName}.onnx or ${baseName}.onnx.gz`)
    const decompressed = await gunzipArrayBuffer(await response.arrayBuffer())
    return await ort.InferenceSession.create(decompressed, sessionOptions)
  }
}

async function gunzipArrayBuffer(buffer: ArrayBuffer): Promise<ArrayBuffer> {
  const Decompression = (self as any).DecompressionStream
  if (!Decompression) {
    throw new Error('This browser cannot decompress gzipped ONNX models.')
  }
  const stream = new Response(buffer).body?.pipeThrough(new Decompression('gzip'))
  if (!stream) throw new Error('Failed to create gzip decompression stream.')
  return await new Response(stream).arrayBuffer()
}

async function loadModels(): Promise<void> {
  post('PROGRESS', { stage: 'Loading ONNX models…', percent: 5 })

  const [kwVocab, stVocab, resources] = await Promise.all([
    fetch(MODEL_BASE + 'keyword_vocab.json').then(r => r.json()),
    fetch(MODEL_BASE + 'structure_vocab.json').then(r => r.json()),
    loadStage3Resources(),
  ])
  kwTokenizer = new WordPieceTokenizer(kwVocab)
  stTokenizer = new WordPieceTokenizer(stVocab)
  stage3Resources = resources

  post('PROGRESS', { stage: 'Loading keyword model…', percent: 15 })

  const numThreads = Math.min(8, self.navigator.hardwareConcurrency || 8)
  ort.env.wasm.numThreads = numThreads
  ort.env.wasm.simd = true
  ort.env.wasm.proxy = false

  pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`

  const sessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    intraOpNumThreads: numThreads,
    interOpNumThreads: 1
  }

  kwSession = await loadOnnxSession('keyword_extractor_int8', sessionOptions)

  post('PROGRESS', { stage: 'Loading structure model…', percent: 40 })

  stSession = await loadOnnxSession('structure_model_int8', sessionOptions)

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
  s_boundary: number
  s_selector: number
  s_bow: number
  s_candidate: number
  s_coverage?: number
  s_rerank?: number
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

    const inputIds: number[] = []
    const attentionMask: number[] = []
    const tokenTypeIds: number[] = []
    const clsPositions: number[] = []
    const tokenMeta: Array<{ sent_idx: number; start: number; end: number } | null> = []

    for (let localIdx = 0; localIdx < windowSentences.length; localIdx++) {
      const globalIdx = winStart + localIdx
      const sent = windowSentences[localIdx]
      const { ids, offsets } = kwTokenizer.encode(sent, MAX_SEQ_LEN - 2)

      const sentIds = ids.slice(1, -1)
      const sentOffsets = offsets.slice(1, -1)

      if (sentIds.length === 0) continue
      if (inputIds.length + sentIds.length + 2 > MAX_SEQ_LEN) break

      const segId = localIdx % 2
      clsPositions.push(inputIds.length)

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
    const bProbs3D: number[][][] = [Array.from({ length: seqLen }, (_, i) => [
      boundaryProbs[i * 3], boundaryProbs[i * 3 + 1], boundaryProbs[i * 3 + 2],
    ])]

    let t = 0
    while (t < seqLen) {
      const meta = tokenMeta[t]
      if (!meta) { t++; continue }
      const bProb = bProbs3D[0][t][1]
      const isB = bProb > 0.5
      if (!isB) { t++; continue }

      let endT = t + 1
      while (
        endT < seqLen &&
        tokenMeta[endT] !== null &&
        (tokenMeta[endT]!).sent_idx === meta.sent_idx &&
        bProbs3D[0][endT][2] > 0.5
      ) {
        endT++
      }

      const spanMeta = tokenMeta[endT - 1]
      if (!spanMeta) { t = endT; continue }

      let boundaryScore = 0
      for (let k = t; k < endT; k++) {
        boundaryScore += Math.max(bProbs3D[0][k][1], bProbs3D[0][k][2])
      }
      boundaryScore /= (endT - t)

      const globalSentIdx = meta.sent_idx
      const sentIdx = globalSentIdx < sentProbs.length ? globalSentIdx : 0

      const surface = sentences[globalSentIdx]
        ? sentences[globalSentIdx].slice(meta.start, spanMeta.end).trim()
        : ''

      if (surface.length >= 3) {
        const selectorScore = sentProbs[Math.min(sentIdx, sentProbs.length - 1)] ?? 0
        allPredictions.push({
          sentence_idx: globalSentIdx,
          start_char: meta.start,
          end_char: spanMeta.end,
          boundary_score: boundaryScore,
          sentence_score: selectorScore,
          surface,
          s_boundary: boundaryScore,
          s_selector: selectorScore,
          s_bow: 0,
          s_candidate: STAGE1_CANDIDATE_WEIGHTS.boundary * boundaryScore
            + STAGE1_CANDIDATE_WEIGHTS.selector * selectorScore,
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

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(1, value))
}

function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9_+\-\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function tokenizePhrase(text: string): string[] {
  const normalized = normalizeText(text)
  return normalized ? normalized.split(/\s+/).filter(Boolean) : []
}

function normalizeSection(section: string | null | undefined): string {
  if (!section) return 'intro'
  const key = section.toLowerCase().replace(/_/g, ' ').trim()
  const aliases: Record<string, string> = {
    abstract: 'intro',
    introduction: 'intro',
    intro: 'intro',
    background: 'intro',
    'related work': 'related_work',
    related_work: 'related_work',
    'prior work': 'related_work',
    methods: 'method',
    method: 'method',
    methodology: 'method',
    experiments: 'experiment',
    experiment: 'experiment',
    evaluation: 'experiment',
    results: 'experiment',
    conclusion: 'conclusion',
    conclusions: 'conclusion',
  }
  return aliases[key] ?? (CANONICAL_SECTIONS.includes(key) ? key : 'intro')
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function normalizedPhraseCount(text: string, phrase: string): number {
  const normalizedText = normalizeText(text)
  const normalizedPhrase = normalizeText(phrase)
  if (!normalizedText || !normalizedPhrase) return 0
  const pattern = new RegExp(`(^|\\s)${escapeRegExp(normalizedPhrase)}(?=\\s|$)`, 'g')
  let count = 0
  while (pattern.exec(normalizedText) !== null) count++
  return count
}

function tokenJaccard(a: string, b: string): number {
  const setA = new Set(tokenizePhrase(a))
  const setB = new Set(tokenizePhrase(b))
  if (setA.size === 0 || setB.size === 0) return 0
  let intersection = 0
  for (const token of setA) if (setB.has(token)) intersection++
  const union = setA.size + setB.size - intersection
  return union === 0 ? 0 : intersection / union
}

function aliasMatchQuality(phrase: string, alias: string): number {
  const normalizedPhrase = normalizeText(phrase)
  const normalizedAlias = normalizeText(alias)
  if (!normalizedPhrase || !normalizedAlias) return 0
  if (normalizedPhrase === normalizedAlias) return 1
  const phraseTokens = new Set(normalizedPhrase.split(/\s+/))
  const aliasTokens = new Set(normalizedAlias.split(/\s+/))
  if (phraseTokens.size === 0 || aliasTokens.size === 0) return 0
  const aliasInPhrase = [...aliasTokens].every(token => phraseTokens.has(token))
  const phraseInAlias = [...phraseTokens].every(token => aliasTokens.has(token))
  if (aliasTokens.size >= 2 && aliasInPhrase) return 0.84
  if (phraseTokens.size >= 2 && phraseInAlias) return 0.78
  const jaccard = tokenJaccard(normalizedPhrase, normalizedAlias)
  return jaccard >= 0.72 ? jaccard : 0
}

function termAliases(term: Stage3BowTerm): string[] {
  const aliases = new Set<string>()
  aliases.add(term.canonical)
  aliases.add(term.display)
  for (const alias of term.aliases ?? []) aliases.add(alias)
  return [...aliases].filter(Boolean)
}

function matchBow(phrase: string, section: string, resources: Stage3Resources): BowMatch | null {
  const targetSection = normalizeSection(section)
  let best: BowMatch | null = null

  for (const term of resources.bow_terms ?? []) {
    if (normalizeSection(term.section) !== targetSection) continue
    for (const alias of termAliases(term)) {
      const quality = aliasMatchQuality(phrase, alias)
      if (quality <= 0) continue
      const support = clamp01((term.confidence ?? 0) * quality)
      if (!best || support > best.bow_support_score) {
        best = {
          term,
          alias,
          match_quality: quality,
          bow_support_score: support,
        }
      }
    }
  }

  return best
}

function buildDocumentTfidf(records: SentenceRecord[], resources: Stage3Resources): Map<string, number> {
  const scores = new Map<string, number>()
  const terms = resources.tfidf_terms ?? []
  if (terms.length === 0) return scores

  for (const term of terms) {
    const section = normalizeSection(term.section)
    const normalizedTerm = normalizeText(term.term)
    if (!normalizedTerm) continue

    let frequency = 0
    for (const record of records) {
      if (normalizeSection(record.section) !== section) continue
      frequency += normalizedPhraseCount(record.text, normalizedTerm)
    }

    if (frequency > 0) {
      const idf = Number.isFinite(term.idf) && term.idf > 0 ? term.idf : 1
      scores.set(`${section}::${normalizedTerm}`, frequency * idf)
    }
  }

  return scores
}

function matchDocumentTfidf(
  phrase: string,
  section: string,
  resources: Stage3Resources,
  documentTfidf: Map<string, number>,
): TfidfMatch | null {
  if (documentTfidf.size === 0) return null
  const targetSection = normalizeSection(section)
  const maxTfidf = Math.max(...documentTfidf.values(), 0)
  if (maxTfidf <= 0) return null

  let best: { term: Stage3TfidfTerm; quality: number; tfidf: number } | null = null
  for (const term of resources.tfidf_terms ?? []) {
    if (normalizeSection(term.section) !== targetSection) continue
    const normalizedTerm = normalizeText(term.term)
    if (!normalizedTerm) continue
    const tfidf = documentTfidf.get(`${targetSection}::${normalizedTerm}`) ?? 0
    if (tfidf <= 0) continue
    const quality = aliasMatchQuality(phrase, normalizedTerm)
    if (quality <= 0) continue
    if (!best || tfidf > best.tfidf || (tfidf === best.tfidf && quality > best.quality)) {
      best = { term, quality, tfidf }
    }
  }

  if (!best) return null
  return {
    section: targetSection,
    term: best.term.term,
    matched_tfidf: best.tfidf,
    max_tfidf: maxTfidf,
    tfidf_support_score: clamp01(best.tfidf / maxTfidf),
  }
}

function buildThresholdTrace(phrase: string, bowMatch: BowMatch | null, tfidfMatch: TfidfMatch | null): ThresholdTrace {
  const phraseWordNumber = tokenizePhrase(phrase).length
  const termConfidence = bowMatch?.term.confidence ?? 0
  const matchQuality = bowMatch?.match_quality ?? 0
  const bowSupportScore = bowMatch?.bow_support_score ?? 0
  const matchedTfidf = tfidfMatch?.matched_tfidf ?? 0
  const maxTfidf = tfidfMatch?.max_tfidf ?? 0
  const tfidfSupportScore = tfidfMatch?.tfidf_support_score ?? 0

  const ngramPassed = phraseWordNumber >= STAGE3_THRESHOLDS.phrase_word_number
  const bowPassed = bowSupportScore >= STAGE3_THRESHOLDS.bow_support_score
  const tfidfPassed = tfidfSupportScore >= STAGE3_THRESHOLDS.tfidf_support_score
  const features = [
    ['ngram_length', ngramPassed],
    ['bow_support', bowPassed],
    ['tfidf_support', tfidfPassed],
  ] as const

  return {
    ngram_length: {
      formula: 'phrase_word_number(u) = |tokenize(p_u)|',
      phrase_word_number: phraseWordNumber,
      threshold: STAGE3_THRESHOLDS.phrase_word_number,
      passed: ngramPassed,
    },
    bow_support: {
      formula: 'bow_support_score(u) = clip_0_1(term_confidence(u) * match_quality(u))',
      term_confidence: clamp01(termConfidence),
      match_quality: clamp01(matchQuality),
      bow_support_score: clamp01(bowSupportScore),
      threshold: STAGE3_THRESHOLDS.bow_support_score,
      passed: bowPassed,
    },
    tfidf_support: {
      formula: 'tfidf_support_score(u) = matched_tfidf(u) / max_tfidf(d)',
      matched_tfidf: matchedTfidf,
      max_tfidf: maxTfidf,
      tfidf_support_score: clamp01(tfidfSupportScore),
      threshold: STAGE3_THRESHOLDS.tfidf_support_score,
      passed: tfidfPassed,
    },
    passing_rule: 'any_feature_passes',
    passed: features.some(([, passed]) => passed),
    passed_features: features.filter(([, passed]) => passed).map(([name]) => name),
    failed_features: features.filter(([, passed]) => !passed).map(([name]) => name),
  }
}

type ScoredCandidate = BIOPrediction & {
  section: string
  bow_match: BowMatch | null
}

function scoreCandidate(pred: BIOPrediction, records: SentenceRecord[]): ScoredCandidate {
  const section = normalizeSection(records[pred.sentence_idx]?.section)
  const bowMatch = matchBow(pred.surface, section, stage3Resources)
  const sBoundary = clamp01(pred.boundary_score)
  const sSelector = clamp01(pred.sentence_score)
  const sBow = clamp01(bowMatch?.bow_support_score ?? 0)
  const sCandidate =
    STAGE1_CANDIDATE_WEIGHTS.boundary * sBoundary +
    STAGE1_CANDIDATE_WEIGHTS.selector * sSelector +
    STAGE1_CANDIDATE_WEIGHTS.bow * sBow

  return {
    ...pred,
    section,
    bow_match: bowMatch,
    s_boundary: sBoundary,
    s_selector: sSelector,
    s_bow: sBow,
    s_candidate: clamp01(sCandidate),
  }
}

function deduplicateCandidates(candidates: ScoredCandidate[]): ScoredCandidate[] {
  const surfaceMap = new Map<string, ScoredCandidate>()
  for (const candidate of candidates) {
    const key = normalizeText(candidate.surface)
    if (!key) continue
    const existing = surfaceMap.get(key)
    if (!existing || candidate.s_candidate > existing.s_candidate) {
      surfaceMap.set(key, candidate)
    }
  }
  return [...surfaceMap.values()].sort((a, b) => b.s_candidate - a.s_candidate)
}

function coverageRerankCandidates(candidates: ScoredCandidate[]): ScoredCandidate[] {
  const selected: ScoredCandidate[] = []
  const pool = [...candidates]

  while (pool.length > 0) {
    let bestIdx = 0
    let bestScore = -Infinity

    for (let i = 0; i < pool.length; i++) {
      const candidate = pool[i]
      const redundancy = selected.reduce(
        (max, item) => Math.max(max, tokenJaccard(candidate.surface, item.surface)),
        0,
      )
      const coverage = 1 - redundancy
      const score =
        STAGE1_RERANK_WEIGHTS.candidate * candidate.s_candidate +
        STAGE1_RERANK_WEIGHTS.coverage * coverage
      if (score > bestScore) {
        bestScore = score
        bestIdx = i
      }
    }

    const chosen = pool.splice(bestIdx, 1)[0]
    const redundancy = selected.reduce(
      (max, item) => Math.max(max, tokenJaccard(chosen.surface, item.surface)),
      0,
    )
    const coverage = 1 - redundancy
    selected.push({
      ...chosen,
      s_coverage: clamp01(coverage),
      s_rerank: clamp01(
        STAGE1_RERANK_WEIGHTS.candidate * chosen.s_candidate +
        STAGE1_RERANK_WEIGHTS.coverage * coverage,
      ),
    })
  }

  return selected
}

function buildConceptUnits(
  sentences: string[],
  records: SentenceRecord[],
  bioPredictions: BIOPrediction[],
  structurePreds: Map<number, SentenceStructure>,
  topK = FINAL_TOP_K,
): ConceptUnit[] {
  const documentTfidf = buildDocumentTfidf(records, stage3Resources)
  const candidatePoolSize = Math.max(topK * CANDIDATE_POOL_MULTIPLIER, topK)
  const deduped = deduplicateCandidates(bioPredictions.map(pred => scoreCandidate(pred, records)))
  const reranked = coverageRerankCandidates(deduped)

  const allUnits = reranked.flatMap((chosen): ConceptUnit[] => {
    const sp = structurePreds.get(chosen.sentence_idx)
    const rec = records[chosen.sentence_idx]
    if (!rec) return []
    const sRerank = clamp01(chosen.s_rerank ?? chosen.s_candidate)
    const evidenceScore = clamp01(sp?.evidence_score ?? 0)
    const sentenceImportance = clamp01(sp?.importance ?? 0)
    const iConcept = clamp01(
      CONCEPT_IMPORTANCE_WEIGHTS.rerank * sRerank +
      CONCEPT_IMPORTANCE_WEIGHTS.evidence * evidenceScore +
      CONCEPT_IMPORTANCE_WEIGHTS.sentence * sentenceImportance,
    )
    const tfidfMatch = matchDocumentTfidf(chosen.surface, rec.section, stage3Resources, documentTfidf)
    const thresholdTrace = buildThresholdTrace(chosen.surface, chosen.bow_match, tfidfMatch)

    return [{
      section: rec.section,
      phrase: chosen.surface,
      role: sp?.role ?? 'none',
      evidence_sentence: sentences[chosen.sentence_idx] ?? '',
      importance: iConcept,
      sentence_index: chosen.sentence_idx,
      s_boundary: chosen.s_boundary,
      s_selector: chosen.s_selector,
      s_bow: chosen.s_bow,
      s_candidate: chosen.s_candidate,
      s_coverage: clamp01(chosen.s_coverage ?? 0),
      s_rerank: sRerank,
      i_concept: iConcept,
      boundary_score: chosen.s_boundary,
      evidence_score: evidenceScore,
      role_score: sp?.role_score ?? 0,
      sentence_importance_score: sentenceImportance,
      threshold_trace: thresholdTrace,
    }]
  })

  const candidatePool = allUnits
    .sort((a, b) => b.i_concept - a.i_concept)
    .slice(0, candidatePoolSize)
  return candidatePool
    .filter(unit => unit.threshold_trace.passed)
    .sort((a, b) => b.i_concept - a.i_concept)
    .slice(0, topK)
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

  // Step B: Section-aware document parsing (upgraded sectioning system)
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
  const units = buildConceptUnits(sentences, records, bioPredictions, structurePreds, FINAL_TOP_K)
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
      await loadModels()
      post('PROGRESS', { stage: 'Models ready', percent: 100 })
    } else if (type === 'INFER') {
      if (!payload?.pdf) throw new Error('No PDF buffer provided')

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
