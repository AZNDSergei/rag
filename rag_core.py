from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
import numpy as np
import faiss
import os

from sentence_transformers import SentenceTransformer

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_OLLAMA = os.getenv("USE_OLLAMA", "0") == "1"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:instruct")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

def load_index(index_dir: str):
    idx_path = Path(index_dir) / "faiss.index"
    meta_path = Path(index_dir) / "meta.jsonl"
    if not idx_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {idx_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.jsonl not found: {meta_path}")
    index = faiss.read_index(str(idx_path))
    metas = [json.loads(l) for l in open(meta_path, "r", encoding="utf-8")]
    emb_model = SentenceTransformer("BAAI/bge-m3")
    return index, metas, emb_model

def embed_query(model, text: str) -> np.ndarray:
    v = model.encode([f"query: {text}"], normalize_embeddings=True).astype("float32")
    return v

def retrieve(index, metas, qvec, k=5) -> List[Dict[str, Any]]:
    D, I = index.search(qvec, k)
    out = []
    for rank, idx in enumerate(I[0], 1):
        m = metas[idx]
        score = 1 - float(D[0][rank-1])   
        out.append({"rank": rank, "score": score, **m})
    return out

FEW_SHOTS = [
    {
        "q": "What documents are required to open an account in Nigeria?",
        "a": "For Nigeria: government-issued ID (passport, national ID, or driver's license), proof of address (utility bill ≤ 3 months), a passport photograph, and a completed application form."
    },
    {
        "q": "I get an error saying my selfie has no face. What should I do?",
        "a": "Retake the selfie with your full face clearly visible and in good lighting, avoiding strong backlight or face coverings."
    }
]

SYSTEM_INSTRUCTION = (
    "You are Ecobank's helpful assistant for account opening and support.\n"
    "Answer strictly based on the provided context passages. If the answer is not in context, say you don't know.\n"
    "You may use the brief recent chat history to disambiguate the user’s intent (country, product, prior clarifications).\n"
    "Provide a SHORT reasoning summary (2–3 bullets) grounded in the context BEFORE the final answer.\n"
    "Do NOT reveal private chain-of-thought; keep the summary concise and user-friendly.\n"
    "Be polite and clear."
)

def _format_history(chat_history: Optional[List[Dict[str, str]]], max_turns: int = 5) -> str:
    """
    chat_history: [{"role": "user"/"assistant", "content": "..."}]
    Возвращает последние max_turns сообщений, от старых к новым.
    """
    if not chat_history:
        return ""
    trimmed = chat_history[-max_turns:]
    lines = []
    for m in trimmed:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)

def build_prompt(
    user_query: str,
    contexts: List[Dict[str, Any]],
    few_shots=FEW_SHOTS,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
  
    ctx = []
    for c in contexts:
        ctx.append(f"[Source {c['rank']} | {c['title']} | {c['doc_path']} | chunk {c['chunk_idx']}]\n{c['text']}")
    context_block = "\n\n".join(ctx)

    shots = "\n\n".join([f"Q: {s['q']}\nA: {s['a']}" for s in few_shots])

    history_block = _format_history(chat_history, max_turns=5)
    if history_block:
        history_block = f"Recent chat history:\n{history_block}\n\n"

    user_block = f"""
{history_block}User question:
{user_query}

Top retrieved context passages:
{context_block}

Few-shot examples (from the same domain):
{shots}

Instructions:
1) Provide a brief, structured reasoning summary (2–3 bullets) grounded in the retrieved context.
2) Then provide the final answer.
3) If the context is insufficient, say "I don't know" and ask for clarifications (e.g., country, product).
""".strip()

    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_block},
    ]

def call_openai_chat(messages, model=None, temperature: float = 0.2) -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)
    model = model or OPENAI_CHAT_MODEL
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return resp.choices[0].message.content

def call_ollama_chat(messages) -> str:
    import requests
    url = "http://localhost:11434/api/chat"
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return (data.get("message") or {}).get("content", "")

def generate_answer(
    user_query: str,
    index_dir: str,
    k: int = 5,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    index, metas, emb_model = load_index(index_dir)
    qvec = embed_query(emb_model, user_query)
    hits = retrieve(index, metas, qvec, k=k)
    messages = build_prompt(user_query, hits, few_shots=FEW_SHOTS, chat_history=chat_history)
    if USE_OLLAMA:
        ans = call_ollama_chat(messages)
    else:
        ans = call_openai_chat(messages)
    return ans, hits
