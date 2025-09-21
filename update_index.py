import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))      
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
MODEL_NAME = os.getenv("EMB_MODEL", "BAAI/bge-m3")

_EMB = None

def log(msg: str, level="INFO"):
    ts = datetime.utcnow().isoformat() + "Z"
    print(f"[{level}] {msg}", flush=True)

def load_model():
    global _EMB
    if _EMB is None:
        log(f"Loading embedding model: {MODEL_NAME}")
        _EMB = SentenceTransformer(MODEL_NAME)
    return _EMB

def iter_files(src: Path) -> List[Path]:
    exts = {".txt", ".md", ".mdx"}
    files = []
    for p in sorted(src.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1", errors="ignore")

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    step = max(1, chunk_size - overlap)
    while i < n:
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        i += step
    return chunks

def embed_chunks(emb_model, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1024), dtype="float32")
    vecs = emb_model.encode(texts, normalize_embeddings=True)
    return vecs.astype("float32")

def atomic_write(path: Path, data_bytes: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data_bytes)
    tmp.replace(path)

def build_full_index(source: Path, out_dir: Path) -> Tuple[int, int]:
    """Full rebuild. Returns (num_files, num_chunks)."""
    emb = load_model()
    files = iter_files(source)
    if not files:
        log("No source files found; will still write an empty index to avoid race at startup.", "WARN")

    all_chunks: List[str] = []
    metas: List[Dict[str, Any]] = []

    file_count = 0
    for f in files:
        text = read_text(f)
        chunks = chunk_text(text)
        for idx, ch in enumerate(chunks):
            metas.append({
                "title": f.name,
                "doc_path": str(f),
                "chunk_idx": idx,
                "text": ch
            })
            all_chunks.append(ch)
        file_count += 1

    if all_chunks:
        vecs = embed_chunks(emb, all_chunks) 
        d = vecs.shape[1]
        index = faiss.IndexFlatIP(d) 
        index.add(vecs)
    else:
        d = 1024
        index = faiss.IndexFlatIP(d)

    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = out_dir / "faiss.index"
    meta_path = out_dir / "meta.jsonl"

    try:
        buf = faiss.serialize_index(index) 
        tmp_path = out_dir / "faiss.index.tmp"
        with open(tmp_path, "wb") as f:
            f.write(buf)
        os.replace(tmp_path, idx_path)
    except Exception as e:
        log(f"Serialize+atomic write failed ({e}); try direct write", "WARN")
        with open(idx_path, "wb") as f:
            f.write(buf)

    meta_bytes = "\n".join(json.dumps(m, ensure_ascii=False) for m in metas).encode("utf-8")
    atomic_write(meta_path, meta_bytes)

    return file_count, len(metas)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Folder with .txt/.md files")
    ap.add_argument("--out", required=True, help="Folder to write faiss.index & meta.jsonl")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    src = Path(args.source).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"START update_index: source={src} out={out_dir}")
    try:
        files, chunks = build_full_index(src, out_dir)
        log(f"FULL REBUILD completed: files={files} chunks={chunks}")
        # small size info
        idx_path = out_dir / "faiss.index"
        meta_path = out_dir / "meta.jsonl"
        size_idx = idx_path.stat().st_size if idx_path.exists() else 0
        size_meta = meta_path.stat().st_size if meta_path.exists() else 0
        log(f"ARTIFACTS: faiss.index={size_idx} bytes, meta.jsonl={size_meta} bytes")
        log("DONE update_index")
        sys.exit(0)
    except Exception as e:
        log(f"Update failed: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
