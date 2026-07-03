import json
import uuid
import time
import numpy as np
import requests
import fitz  # type: ignore
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from KG_generation import (
    build_knowledge_graph_from_texts,
    format_bow_terms_for_prompt,
    match_bow_terms_in_texts,
)
from rag import rank_evidence_units

import os
import base64
from abc import ABC, abstractmethod
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
CORS(app)

LLM_MAX_TOKENS: int = 4096
LLM_TEMPERATURE: float = 0.3

# ── Ollama local settings ─────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL: str = "mistral:latest"

REQUEST_TIMEOUT = 30  # seconds
EMBED_BATCH_SIZE = 50  # chunks per embedding request
TTL_SECONDS = 3600  # 1-hour session expiry

BASE_DIR = Path(__file__).parent.resolve()
VECTOR_STORE_PATH = BASE_DIR / "vector_store.json"
DOC_STORE_PATH = BASE_DIR / "doc_store.json"

AUTH_TOKEN = os.environ.get("MISTRAL_TOKEN", "")
MISTRAL_CHAT_MODEL: str = "mistral-small-latest"
EMBEDDING_ENDPOINT = "aHR0cHM6Ly9hcGkubWlzdHJhbC5haS92MS9lbWJlZGRpbmdz"
CHAT_ENDPOINT = "aHR0cHM6Ly9hcGkubWlzdHJhbC5haS92MS9jaGF0L2NvbXBsZXRpb25z"


class LLMBackend(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], session: requests.Session | None = None) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class MistralBackend(LLMBackend):
    @property
    def name(self) -> str:
        return f"mistral-cloud/{MISTRAL_CHAT_MODEL}"

    def chat(self, messages: list[dict], session: requests.Session | None = None) -> str:
        if not AUTH_TOKEN:
            raise RuntimeError(
                "Missing MISTRAL_TOKEN. Set it via the environment variable."
            )
        headers = {
            "Authorization": f"Bearer {AUTH_TOKEN}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": MISTRAL_CHAT_MODEL,
            "messages": messages,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "stream": False,
        }
        req = session or requests
        resp = req.post(
            base64.b64decode(CHAT_ENDPOINT).decode("utf-8"),
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected Mistral response: {data}") from exc


class OllamaBackend(LLMBackend):
    @property
    def name(self) -> str:
        return f"ollama/{OLLAMA_CHAT_MODEL}"

    def chat(self, messages: list[dict], session: requests.Session | None = None) -> str:
        payload: dict[str, Any] = {
            "model": OLLAMA_CHAT_MODEL,
            "messages": messages,
            "options": {
                "num_predict": LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE,
            },
            "stream": False,
        }
        req = session or requests
        resp = req.post(
            f"{OLLAMA_BASE_URL}{base64.b64decode('L2FwaS9jaGF0').decode('utf-8')}",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response: {data}") from exc


_llm_executor = ThreadPoolExecutor(max_workers=10)

def llm_chat(messages: list[dict]) -> tuple[str, str]:
    backends = [MistralBackend(), OllamaBackend()]
    sessions = {b: requests.Session() for b in backends}
    
    future_to_backend = {
        _llm_executor.submit(b.chat, messages, sessions[b]): b for b in backends
    }
    
    try:
        for future in as_completed(future_to_backend):
            backend = future_to_backend[future]
            try:
                reply = future.result()
                return reply, backend.name
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(f"Backend {backend.name} failed: {exc}")
                
        raise RuntimeError("All LLM backends failed to produce a response.")
    finally:
        for session in sessions.values():
            session.close()


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
    """Passive TTL garbage collection — triggered on each /upload call."""
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


# ─── Embedding Service ────────────────────────────────────────────────────────────


def _embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if not AUTH_TOKEN:
        raise RuntimeError("Missing AUTH_TOKEN. Set it via the environment variable.")

    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {"model": "mistral-embed", "input": texts}

    resp = requests.post(
        base64.b64decode(EMBEDDING_ENDPOINT).decode("utf-8"),
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
    text_lower = text.lower()
    matched = [kw for kw in keywords if kw in text_lower]
    ratio = len(matched) / max(len(keywords), 1)
    return ratio, matched


def build_system_prompt_from_evidence_units(
    evidence_units: list[dict],
    section_notes: dict,
    title: str,
    bow_terms: list[dict] | None = None,
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

    bow_context = format_bow_terms_for_prompt(bow_terms or [])

    return (
        "You are an expert scientific paper analysis assistant with deep expertise in NLP and machine learning research.\n"
        "The following structured analysis was automatically extracted from the paper using a SciBERT-based Evidence-grounded\n"
        "Concept Unit extraction system. Each concept unit contains: the key phrase, its rhetorical role (e.g. core_method,\n"
        "result, contribution), an importance score (0–1), and the supporting evidence sentence.\n\n"
        "Use this structured knowledge to answer the user's question with precision and specificity. "
        "Cite specific concept units and evidence sentences when relevant. "
        "When matched terminology is provided, include short explanations of the relevant technical terms before or within "
        "the paper interpretation, and connect each term back to how it appears in this paper. "
        "If the answer cannot be found in the provided context, say that:"
        "No evidence was extracted for this part from the provided paper analysis."
        "Do not use prior knowledge, the paper title, or general knowledge to fill missing sections.\n\n"
        f"=== Paper: {title} ===\n\n"
        f"{structured_context}"
        f"{bow_context}"
    )


def build_system_prompt_from_chunks(
    retrieved_context: str,
    bow_terms: list[dict] | None = None,
) -> str:
    """
    Step B (Server fallback): Classic RAG prompt using raw text chunks.
    """
    bow_context = format_bow_terms_for_prompt(bow_terms or [])
    return (
        "You are a scientific paper analysis assistant. "
        "Answer the user's question based strictly and only on the provided context. "
        "When matched terminology is provided, include concise explanations of the relevant technical terms and connect "
        "them to the retrieved paper passages. "
        "If the answer cannot be found in the context, say that:"
        "No evidence was extracted for this part from the provided paper analysis."
        "Do not use prior knowledge, the paper title, or general knowledge to fill missing sections.\n\n"
        f"Context from paper:\n{retrieved_context}"
        f"{bow_context}"
    )


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    Returns service readiness and which inference modes are available.
    """
    return jsonify(
        {
            "status": "ok",
            "token_configured": bool(AUTH_TOKEN),
            "llm_backend": "hybrid",
            "modes": {
                "edge": True,  # accepts evidence_units payloads
                "server": True,  # accepts doc_id payloads (classic RAG)
            },
            "version": "2.0.0",
        }
    )


@app.route("/upload", methods=["POST"])
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

    if not AUTH_TOKEN:
        return jsonify({"error": "Server not configured"}), 503

    try:
        # 1. PyMuPDF PDF parse
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

        # 3. Batch embedding
        try:
            all_embeddings = fetch_remote_embeddings_batch(chunks)
        except requests.exceptions.Timeout:
            return jsonify(
                {"error": "Embedding request timed out. Please try again."}
            ), 504
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Embedding request error: {e}"}), 502

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
        app.logger.exception("Unhandled error in /upload")
        return jsonify({"error": "Internal server error during PDF processing."}), 500


@app.route("/chat", methods=["POST"])
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
    bow_terms: list[dict] = []
    knowledge_graph: dict = {"terms": [], "edges": []}

    if evidence_units:
        # ── Mode A: Edge inference path ──────────────────────────────────
        edge_rag = rank_evidence_units(
            evidence_units=evidence_units,
            query_keywords=query_keywords,
            prompt_top_k=14,
            display_top_k=5,
        )
        selected_evidence_units = (
            edge_rag["prompt_evidence_units"] or evidence_units[:14]
        )
        context_chunks = edge_rag["context_chunks"]
        knowledge_graph = edge_rag["knowledge_graph"]
        bow_terms = knowledge_graph["terms"]
        system_prompt = build_system_prompt_from_evidence_units(
            evidence_units=selected_evidence_units,
            section_notes=section_notes,
            title=title,
            bow_terms=bow_terms,
        )
        app.logger.info(
            f"[Chat/Edge] {len(evidence_units)} units → {len(selected_evidence_units)} retrieved, {len(bow_terms)} BoW terms, {len(knowledge_graph['edges'])} KG edges, title='{title}', query length={len(user_query)}"
        )

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
            return jsonify(
                {"error": "Embedding request timed out. Please try again."}
            ), 504
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Embedding request error: {e}"}), 502

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
        knowledge_graph = build_knowledge_graph_from_texts(
            context_parts,
            term_limit=12,
            edge_limit=24,
        )
        bow_terms = knowledge_graph["terms"]
        for chunk in context_chunks:
            chunk["matched_bow_terms"] = match_bow_terms_in_texts(
                [chunk["text"]], limit=5
            )

        system_prompt = build_system_prompt_from_chunks(
            retrieved_context,
            bow_terms=bow_terms,
        )
        app.logger.info(
            f"[Chat/Server] doc_id={doc_id[:8]}…, retrieved {len(context_parts)} chunks, {len(bow_terms)} BoW terms, {len(knowledge_graph['edges'])} KG edges"
        )

    else:
        return jsonify(
            {
                "error": "Must provide either 'evidence_units' (edge mode) or 'doc_id' (server mode)"
            }
        ), 400

    # ── Build chat messages ────────────────────────────────────────────────
    chat_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if msg.get("role") in ("user", "assistant"):
            chat_messages.append({"role": msg["role"], "content": msg["content"]})

    # ── Call Hybrid LLM ───────────────────────────
    try:
        reply, used_backend = llm_chat(chat_messages)
        app.logger.info(f"[Chat] Used backend: {used_backend}")
        return jsonify(
            {
                "reply": reply,
                "context_chunks": context_chunks,
                "query_keywords": query_keywords,
                "term_explanations": bow_terms,
                "knowledge_graph": knowledge_graph,
                "used_backend": used_backend,
            }
        )

    except Exception as e:
        app.logger.error(f"Hybrid LLM inference failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if not AUTH_TOKEN:
        print("WARNING: MISTRAL_TOKEN not set. Set it via environment variable:")
        print("   export MISTRAL_TOKEN='key'")
    app.run(host="0.0.0.0", port=5000, debug=False)
