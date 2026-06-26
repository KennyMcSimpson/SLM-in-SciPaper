"""
SciPaper RAG - Backend Gateway
================================
Step 3 of the Client-Edge-Cloud migration: the backend is now a pure LLM
proxy / API gateway with ~20 MB memory footprint.

Supported modes:
  A. Edge mode (Client inference already done):
     POST /api/chat  {evidence_units, section_notes, title, messages}
     → Assembles a structured Prompt from Evidence JSON and proxies to Mistral.

  B. Server fallback (classic RAG, for environments without ONNX models):
     POST /api/upload  {file: PDF}
     → PyMuPDF parse → Mistral Embed → store in JSON vector DB
     POST /api/chat  {doc_id, messages}
     → Cosine retrieval → Mistral Chat

  GET /api/health  → status + current mode capabilities

Both modes share the same /api/chat endpoint; the backend detects which
payload shape was received and routes accordingly.
"""

import json
import uuid
import time
import numpy as np
import requests
import fitz  # PyMuPDF
from pathlib import Path
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── Configuration ────────────────────────────────────────────────────────────
import os

API_KEY = os.environ.get("MISTRAL_API_KEY", "")
EMBEDDING_API_URL = "https://api.mistral.ai/v1/embeddings"
CHAT_API_URL = "https://api.mistral.ai/v1/chat/completions"

REQUEST_TIMEOUT = 30  # seconds
EMBED_BATCH_SIZE = 50  # chunks per embedding request
TTL_SECONDS = 3600  # 1-hour session expiry

BASE_DIR = Path(__file__).parent.resolve()
VECTOR_STORE_PATH = BASE_DIR / "vector_store.json"
DOC_STORE_PATH = BASE_DIR / "doc_store.json"


# ─── Storage Helpers ─────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict:
    """Safe JSON loader; returns {} on missing or corrupt file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def cleanup_expired_docs() -> None:
    """Passive TTL garbage collection — triggered on each /api/upload call."""
    now = time.time()
    docs = _load_json(DOC_STORE_PATH)
    vectors = _load_json(VECTOR_STORE_PATH)

    expired = [
        did
        for did, data in docs.items()
        if now - data.get("_timestamp", now) > TTL_SECONDS
    ]
    if not expired:
        return

    for did in expired:
        docs.pop(did, None)
        vectors.pop(did, None)

    _save_json(DOC_STORE_PATH, docs)
    _save_json(VECTOR_STORE_PATH, vectors)
    app.logger.info(f"[TTL] Cleaned up {len(expired)} expired session(s).")


# ─── Text Processing ──────────────────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Character-level text chunker with sentence-boundary alignment.
    Mirrors the same logic used in the frontend worker for consistency.
    """
    chunks: list[str] = []
    text = text.strip()
    total = len(text)
    start = 0

    while start < total:
        end = min(start + chunk_size, total)
        if end < total:
            period = text.find(".", end)
            if 0 < period - end < 150:
                end = period + 1

        chunk = text[start:end].strip()
        if len(chunk) > 20:
            chunks.append(chunk)

        next_start = end - overlap
        if next_start <= start:
            next_start = start + chunk_size
        start = next_start

    return chunks


# ─── Embedding API ────────────────────────────────────────────────────────────


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Single Mistral embedding API call for up to EMBED_BATCH_SIZE texts."""
    if not texts:
        return []
    if not API_KEY:
        raise RuntimeError("API_KEY not configured.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": "mistral-embed", "input": texts}

    resp = requests.post(
        EMBEDDING_API_URL,
        json=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


def fetch_remote_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch-calls _embed_batch and aggregates results."""
    results: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        results.extend(_embed_batch(texts[i : i + EMBED_BATCH_SIZE]))
    return results


# ─── Retrieval ────────────────────────────────────────────────────────────────


def compute_matches(
    vector_store: dict, query_vector: list[float], top_k: int = 5
) -> list[tuple[str, str, float]]:
    """
    Pure NumPy cosine similarity retrieval.
    Returns list of (doc_id, chunk_id, score) triples, sorted by score desc.
    """
    q = np.array(query_vector, dtype=np.float32)
    norm_q = float(np.linalg.norm(q))
    if norm_q == 0:
        return []

    scored: list[tuple[str, str, float]] = []

    for doc_id, chunks in vector_store.items():
        for chunk_id, embedding in chunks.items():
            if chunk_id.startswith("_"):  # skip metadata fields
                continue
            c = np.array(embedding, dtype=np.float32)
            norm_c = float(np.linalg.norm(c))
            if norm_c == 0:
                continue
            score = float(np.dot(q, c) / (norm_q * norm_c))
            scored.append((doc_id, chunk_id, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


# ─── BoW / Keyword Extraction (for highlighting) ────────────────────────────

IMPORTANT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "do",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "we",
    "with",
    "which",
    "what",
    "how",
    "when",
    "where",
    "who",
    "will",
    "would",
    "not",
    "also",
    "but",
    "if",
    "then",
    "than",
    "so",
    "yet",
    "more",
    "most",
}


def extract_query_keywords(query: str, max_kw: int = 12) -> list[str]:
    """
    Extract meaningful content words from the user query for BoW highlighting.
    Returns lowercase single tokens and 2-grams sorted by length (longer first).
    """
    import re

    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-']{1,}", query.lower())
    content_words = [t for t in tokens if t not in IMPORTANT_STOPWORDS and len(t) > 2]
    # Build 2-grams for phrase matching
    bigrams = [
        f"{content_words[i]} {content_words[i + 1]}"
        for i in range(len(content_words) - 1)
    ]
    # Combine: bigrams first (for better match specificity), then unigrams
    all_kw = list(dict.fromkeys(bigrams + content_words))  # preserve order, deduplicate
    return all_kw[:max_kw]


def score_text_vs_keywords(text: str, keywords: list[str]) -> tuple[float, list[str]]:
    """
    Returns (match_ratio, matched_keywords) for a piece of text vs keyword list.
    Uses simple substring matching (case-insensitive).
    """
    text_lower = text.lower()
    matched = [kw for kw in keywords if kw in text_lower]
    ratio = len(matched) / max(len(keywords), 1)
    return ratio, matched


# ─── Prompt Builders ─────────────────────────────────────────────────────────


def build_system_prompt_from_evidence_units(
    evidence_units: list[dict],
    section_notes: dict,
    title: str,
) -> str:
    """
    Step 3 (Edge mode): Assembles a rich structured Prompt from the
    Evidence-grounded Concept Unit JSON produced by the browser inference worker.

    Groups units by section, annotates each with its role and importance score,
    and prepends high-level section summaries from section_notes.
    """
    # Group units by section in canonical order
    SECTION_ORDER = ["intro", "related_work", "method", "experiment", "conclusion"]
    SECTION_LABELS = {
        "intro": "Introduction",
        "related_work": "Related Work",
        "method": "Methodology",
        "experiment": "Experiments & Results",
        "conclusion": "Conclusion",
    }

    units_by_section: dict[str, list[dict]] = {}
    for unit in evidence_units:
        sec = unit.get("section", "intro")
        units_by_section.setdefault(sec, []).append(unit)

    sections_text: list[str] = []

    for sec in SECTION_ORDER:
        units = units_by_section.get(sec)
        notes = section_notes.get(sec, [])
        if not units and not notes:
            continue

        label = SECTION_LABELS.get(sec, sec.replace("_", " ").title())
        lines: list[str] = [f"### {label}"]

        if notes:
            lines.append("**Key statements:**")
            for note in notes[:3]:
                lines.append(f"  - {note}")

        if units:
            lines.append("**Concept units (phrase → role → evidence):**")
            # Sort by importance descending, show top-10 per section
            sorted_units = sorted(
                units, key=lambda u: u.get("importance", 0), reverse=True
            )
            for u in sorted_units[:10]:
                phrase = u.get("phrase", "")
                role = u.get("role", "none").replace("_", " ")
                evidence = u.get("evidence_sentence", "")
                importance = u.get("importance", 0)
                lines.append(
                    f"  - [{role.upper()}] **{phrase}** (importance: {importance:.2f})\n"
                    f'    Evidence: "{evidence[:200]}"'
                )

        sections_text.append("\n".join(lines))

    structured_context = (
        "\n\n".join(sections_text)
        if sections_text
        else "No structured context available."
    )

    return (
        "You are an expert scientific paper analysis assistant with deep expertise in NLP and machine learning research.\n"
        "The following structured analysis was automatically extracted from the paper using a SciBERT-based Evidence-grounded\n"
        "Concept Unit extraction system. Each concept unit contains: the key phrase, its rhetorical role (e.g. core_method,\n"
        "result, contribution), an importance score (0–1), and the supporting evidence sentence.\n\n"
        "Use this structured knowledge to answer the user's question with precision and specificity. "
        "Cite specific concept units and evidence sentences when relevant. "
        "If the answer cannot be found in the provided context, say so clearly.\n\n"
        f"=== Paper: {title} ===\n\n"
        f"{structured_context}"
    )


def build_system_prompt_from_chunks(retrieved_context: str) -> str:
    """
    Step B (Server fallback): Classic RAG prompt using raw text chunks.
    """
    return (
        "You are a scientific paper analysis assistant. "
        "Answer the user's question based strictly and only on the provided context. "
        "If the answer cannot be found in the context, say so clearly.\n\n"
        f"Context from paper:\n{retrieved_context}"
    )


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.route("/api/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    Returns API readiness and which inference modes are available.
    """
    api_ready = bool(API_KEY)
    return jsonify(
        {
            "status": "ok",
            "api_key_configured": api_ready,
            "modes": {
                "edge": True,  # always accepts evidence_units payloads
                "server": True,  # always accepts doc_id payloads (classic RAG)
            },
            "version": "2.0.0",
        }
    )


@app.route("/api/upload", methods=["POST"])
def upload_paper():
    """
    Server-fallback upload endpoint (Mode B / classic RAG path).
    Accepts a PDF file → PyMuPDF parse → Mistral Embed → JSON vector DB.
    Returns doc_id + preview chunks.

    In Edge mode (Mode A), this endpoint is never called — the browser
    handles PDF parsing and inference locally.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    if not API_KEY:
        return jsonify({"error": "Server not configured"}), 503

    try:
        # 1. PyMuPDF PDF parse (< 30 MB memory, milliseconds)
        pdf_bytes = file.read()
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = "\n".join(page.get_text("text") for page in pdf_doc)
        pdf_doc.close()

        if not full_text.strip():
            return jsonify({"error": "PDF appears to be empty or image-only"}), 400

        # 2. Text chunking
        chunks = chunk_text(full_text, chunk_size=800, overlap=100)
        if not chunks:
            return jsonify({"error": "Could not extract text from PDF"}), 400

        # 3. Batch embedding (50 chunks / request, with timeout)
        try:
            all_embeddings = fetch_remote_embeddings_batch(chunks)
        except requests.exceptions.Timeout:
            return jsonify({"error": "Embedding API timed out. Please try again."}), 504
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Embedding API error: {e}"}), 502

        # 4. Build session data
        doc_id = str(uuid.uuid4())
        ts = time.time()
        new_doc_entry: dict = {"_timestamp": ts}
        new_vec_entry: dict = {"_timestamp": ts}

        for idx, (chunk, emb) in enumerate(zip(chunks, all_embeddings)):
            cid = f"chunk_{idx}"
            new_doc_entry[cid] = {"text": chunk}
            new_vec_entry[cid] = emb

        # 5. Persist (with passive TTL cleanup)
        cleanup_expired_docs()
        docs = _load_json(DOC_STORE_PATH)
        vecs = _load_json(VECTOR_STORE_PATH)
        docs[doc_id] = new_doc_entry
        vecs[doc_id] = new_vec_entry
        _save_json(DOC_STORE_PATH, docs)
        _save_json(VECTOR_STORE_PATH, vecs)

        # 6. Return doc_id + preview
        preview = [
            {"role": "context", "text": chunks[i]} for i in range(min(5, len(chunks)))
        ]
        return jsonify(
            {
                "doc_id": doc_id,
                "chunk_count": len(chunks),
                "preview": preview,
            }
        )

    except Exception:
        app.logger.exception("Unhandled error in /api/upload")
        return jsonify({"error": "Internal server error during PDF processing."}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Unified chat endpoint supporting both inference modes.

    Mode A (Edge inference — preferred):
      Expects: { messages, evidence_units, section_notes, title }
      Action: Assembles structured prompt from Evidence JSON → Mistral Chat.
      Memory: ~0 MB (no retrieval, no vector store access).

    Mode B (Server-side RAG — fallback):
      Expects: { messages, doc_id }
      Action: Embeds query → cosine retrieval → Mistral Chat.
      Memory: minimal (NumPy cosine similarity only).
    """
    data = request.get_json(silent=True) or {}
    messages: list[dict] = data.get("messages", [])

    if not messages:
        return jsonify({"error": "Missing messages"}), 400
    if not API_KEY:
        return jsonify({"error": "Server not configured"}), 503

    # Extract latest user message
    user_query = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    if not user_query:
        return jsonify({"error": "No user message found"}), 400

    # ── Route by payload shape ──────────────────────────────────────────────
    evidence_units: list[dict] = data.get("evidence_units", [])
    section_notes: dict = data.get("section_notes", {})
    title: str = data.get("title", "Scientific Paper")
    doc_id: str = data.get("doc_id", "")

    # Extract BoW keywords from user query for frontend highlighting
    query_keywords = extract_query_keywords(user_query)

    # context_chunks is the structured visualization payload returned to frontend
    context_chunks: list[dict] = []

    if evidence_units:
        # ── Mode A: Edge inference path ──────────────────────────────────
        system_prompt = build_system_prompt_from_evidence_units(
            evidence_units=evidence_units,
            section_notes=section_notes,
            title=title,
        )
        app.logger.info(
            f"[Chat/Edge] {len(evidence_units)} units, title='{title}', query length={len(user_query)}"
        )

        # Build context_chunks: score each unit's evidence sentence vs query keywords
        # Return top-5 most query-relevant units for visualization
        SECTION_ORDER = ["intro", "related_work", "method", "experiment", "conclusion"]
        SECTION_LABELS = {
            "intro": "Introduction",
            "related_work": "Related Work",
            "method": "Methodology",
            "experiment": "Experiments & Results",
            "conclusion": "Conclusion",
        }
        scored_units: list[tuple[float, dict]] = []
        for unit in evidence_units:
            ev_text = unit.get("evidence_sentence", "")
            phrase = unit.get("phrase", "")
            combined_text = f"{phrase} {ev_text}"
            kw_score, matched_kw = score_text_vs_keywords(combined_text, query_keywords)
            importance = unit.get("importance")
            if importance is None:
                importance = 0.0
            importance = float(importance)
            # Combined relevance: 60% keyword match, 40% model importance
            relevance = 0.6 * kw_score + 0.4 * importance
            scored_units.append(
                (
                    relevance,
                    {
                        "text": ev_text,
                        "phrase": phrase,
                        "section": unit.get("section", "intro"),
                        "section_label": SECTION_LABELS.get(
                            unit.get("section", "intro"), "Paper"
                        ),
                        "role": unit.get("role", "none"),
                        "score": round(relevance, 4),
                        "importance": round(importance, 4),
                        "boundary_score": round(unit.get("boundary_score", 0.0), 4),
                        "evidence_score": round(unit.get("evidence_score", 0.0), 4),
                        "matched_keywords": matched_kw,
                        "source": "edge",
                    },
                )
            )

        # Sort by relevance, return top-5
        scored_units.sort(key=lambda x: x[0], reverse=True)
        context_chunks = [item for _, item in scored_units[:5]]

    elif doc_id:
        # ── Mode B: Server-side RAG path ─────────────────────────────────
        vecs = _load_json(VECTOR_STORE_PATH)
        docs = _load_json(DOC_STORE_PATH)

        if doc_id not in vecs or doc_id not in docs:
            return jsonify(
                {"error": "Document session not found or expired. Please re-upload."}
            ), 404

        # Embed the user query
        try:
            query_vec = fetch_remote_embeddings_batch([user_query])[0]
        except requests.exceptions.Timeout:
            return jsonify({"error": "Embedding API timed out. Please try again."}), 504
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Embedding API error: {e}"}), 502

        # Retrieve top-5 chunks with scores (isolated to this session's doc_id)
        user_vecs = {doc_id: vecs[doc_id]}
        matches = compute_matches(user_vecs, query_vec, top_k=5)

        context_parts = []
        for rank, (d_id, c_id, score) in enumerate(matches):
            chunk_data = docs.get(d_id, {}).get(c_id)
            if chunk_data and "text" in chunk_data:
                chunk_text_val = chunk_data["text"]
                context_parts.append(chunk_text_val)
                _, matched_kw = score_text_vs_keywords(chunk_text_val, query_keywords)
                context_chunks.append(
                    {
                        "text": chunk_text_val,
                        "phrase": None,  # no keyphrase in server mode
                        "section": None,
                        "section_label": f"Chunk {rank + 1}",
                        "role": None,
                        "score": round(float(score), 4),
                        "importance": None,
                        "boundary_score": None,
                        "evidence_score": None,
                        "matched_keywords": matched_kw,
                        "source": "server",
                    }
                )

        retrieved_context = (
            "\n\n---\n\n".join(context_parts)
            if context_parts
            else "No relevant context found in the document."
        )
        system_prompt = build_system_prompt_from_chunks(retrieved_context)
        app.logger.info(
            f"[Chat/Server] doc_id={doc_id[:8]}…, retrieved {len(context_parts)} chunks"
        )

    else:
        return jsonify(
            {
                "error": "Must provide either 'evidence_units' (edge mode) or 'doc_id' (server mode)"
            }
        ), 400

    # ── Build API messages ────────────────────────────────────────────────
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if msg.get("role") in ("user", "assistant"):
            api_messages.append({"role": msg["role"], "content": msg["content"]})

    # ── Call Mistral Chat ─────────────────────────────────────────────────
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mistral-small-latest",
        "messages": api_messages,
        "max_tokens": 1000,  # slightly higher for structured evidence responses
        "temperature": 0.3,
    }

    try:
        resp = requests.post(
            CHAT_API_URL,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        return jsonify(
            {
                "reply": reply,
                "context_chunks": context_chunks,
                "query_keywords": query_keywords,
            }
        )

    except requests.exceptions.Timeout:
        return jsonify({"error": "Chat API timed out. Please try again."}), 504
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Chat API error: {e}")
        return jsonify({"error": f"Chat API error: {e}"}), 502
    except (KeyError, IndexError) as e:
        app.logger.error(f"Unexpected Chat API response format: {e}")
        return jsonify({"error": "Unexpected response from AI service."}), 500


if __name__ == "__main__":
    if not API_KEY:
        print("WARNING: MISTRAL_API_KEY not set. Set it via environment variable:")
        print("   export MISTRAL_API_KEY='key'")
    app.run(host="0.0.0.0", port=5000, debug=False)
