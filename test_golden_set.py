import argparse, json, os, time, unicodedata, re, difflib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
load_dotenv()

from rag_core import generate_answer

OK_PHRASES_UNKNOWN = [
    "i don't know", "i do not know",
    "not in the provided context", "insufficient context",
    "cannot find", "no information found",
    "je ne sais pas", "je n’en sais pas", "je n en sais pas"
]

NUM_WORDS = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
    "eleven":"11","twelve":"12","thirteen":"13","fourteen":"14","fifteen":"15","sixteen":"16","seventeen":"17","eighteen":"18","nineteen":"19",
    "twenty":"20"
}

def ensure_logs(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def append_jsonl(path: Path, rec: dict):
    ensure_logs(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def to_ascii_dash(s: str) -> str:
    return s.replace("–","-").replace("—","-").replace("−","-")

def normalize_numbers_tokens(s: str) -> str:
    s = " ".join(NUM_WORDS.get(tok, tok) for tok in s.split())
    s = re.sub(r"\b(\d+)\s*(?:to|–|—|-)\s*(\d+)\b", r"\1-\2", s)
    return s

def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = unicodedata.normalize("NFKC", s)
    s = to_ascii_dash(s)
    s = normalize_numbers_tokens(s)
    s = re.sub(r"\s+", " ", s)
    return s

def fuzzy_contains(needle: str, hay: str, threshold: float=0.78) -> bool:
    n = norm(needle); h = norm(hay)
    if not n: return True
    if n in h: return True
    ratio = difflib.SequenceMatcher(None, n, h).ratio()
    if ratio >= threshold: return True
    words = n.split()
    if len(words) >= 3:
        win = max(3, min(8, len(words)))
        h_words = h.split()
        for i in range(0, max(1, len(h_words)-win+1)):
            chunk = " ".join(h_words[i:i+win])
            if difflib.SequenceMatcher(None, n, chunk).ratio() >= threshold:
                return True
    return False

def any_keyword_match(keywords: List[str], text: str) -> bool:
    return any(fuzzy_contains(k, text) for k in (keywords or []) if k)

def contains_unknown_phrase(answer: str) -> bool:
    t = norm(answer)
    return any(p in t for p in OK_PHRASES_UNKNOWN)

def sources_list(hits: List[Dict[str, Any]]) -> List[Dict[str,str]]:
    out = []
    for h in hits:
        doc_path = h.get("doc_path") or ""
        title = h.get("title") or ""
        stem = Path(doc_path).stem if doc_path else ""
        out.append({"stem": stem, "title": title, "path": doc_path})
    return out

def match_any_source(expected_stems: List[str], actual: List[Dict[str,str]]) -> bool:
    if not expected_stems:
        return True
    exp = [norm(x) for x in expected_stems]
    for a in actual:
        stem = norm(a.get("stem",""))
        title = norm(a.get("title",""))
        path = norm(a.get("path",""))
        for e in exp:
            if e and (e in stem or stem in e or e in title or e in path):
                return True
    return False

def eval_positive(answer: str, hits: List[Dict[str,Any]],
                  must_have_any: List[str], expect_sources_any: List[str],
                  strict: bool) -> Tuple[bool, dict]:
    srcs = sources_list(hits)
    ok_text = True if not must_have_any else any_keyword_match(must_have_any, answer)
    ok_src  = match_any_source(expect_sources_any, srcs)
    ok = (ok_text and ok_src) if strict else (ok_text or ok_src)
    return ok, {"ok_text": ok_text, "ok_src": ok_src, "srcs": srcs}

def eval_negative(answer: str, hits: List[Dict[str,Any]]) -> Tuple[bool, dict]:
    unknown = contains_unknown_phrase(answer)
    srcs = sources_list(hits)
    ok = unknown or len(srcs) == 0
    return ok, {"unknown_phrase": unknown, "srcs": srcs}

def run_set(items: List[dict], label: str, strict: bool, debug: bool,
            index_dir: str, log_path: Path) -> Tuple[int,int]:
    passed, total = 0, len(items)
    print(f"\n--- Running {label}: {total} items ---")
    for i, it in enumerate(items, 1):
        q = it["q"]
        t0 = time.time()
        try:
            ans, hits = generate_answer(q, index_dir=index_dir, k=5)
        except Exception as e:
            ans, hits = f"[ERROR] {e}", []
        elapsed = round(time.time() - t0, 3)

        if label == "positive":
            ok, details = eval_positive(ans, hits, it.get("must_have_any", []), it.get("expect_sources_any", []), strict)
        else:
            ok, details = eval_negative(ans, hits)

        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "label": label, "idx": i, "q": q,
            "latency_sec": elapsed,
            "found_chunks": len(hits) > 0,
            "answer_len": len(ans or ""),
            "success": ok,
            "details": details,
            "sources": [
                {
                    "rank": h.get("rank"),
                    "title": h.get("title"),
                    "doc_path": h.get("doc_path"),
                    "chunk_idx": h.get("chunk_idx"),
                    "score": h.get("score"),
                } for h in hits
            ],
            "answer_preview": (ans or "")[:600],
        }
        append_jsonl(log_path, rec)

        status = "OK" if ok else "FAIL"
        print(f"[{label}] {i}/{total} {status}  {q}  ({elapsed}s)")
        if debug and not ok:
            print("  ├─ must_have_any:", it.get("must_have_any", []))
            print("  ├─ expect_sources_any:", it.get("expect_sources_any", []))
            print("  ├─ src stems:", [s['stem'] for s in details['srcs']])
            print("  └─ answer:", (ans or "")[:600].replace("\n","  "))
        passed += 1 if ok else 0
    return passed, total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="index_out")
    ap.add_argument("--golden", default="golden_set.json")
    ap.add_argument("--log", default="logs/golden_results.jsonl")
    ap.add_argument("--strict", action="store_true", help="require both text and source to match for positives")
    ap.add_argument("--debug", action="store_true", help="print mismatch diagnostics")
    ap.add_argument("--only", choices=["pos","neg","both"], default="both")
    args = ap.parse_args()

    log_path = Path(args.log)
    gs = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    pos = gs.get("positives", [])
    neg = gs.get("negatives", []) or gs.get("negative", [])  # на всякий случай

    print(f"Golden file: {args.golden}")
    print(f"Index dir:   {args.index_dir}")
    print(f"Loaded:      positives={len(pos)}  negatives={len(neg)}  strict={args.strict}")

    p_ok = p_total = n_ok = n_total = 0

    if args.only in ("pos", "both"):
        p_ok, p_total = run_set(pos, "positive", args.strict, args.debug, args.index_dir, log_path)
    if args.only in ("neg", "both"):
        n_ok, n_total = run_set(neg, "negative", args.strict, args.debug, args.index_dir, log_path)

    print("\n=== SUMMARY ===")
    print(f"Positives: {p_ok}/{p_total} passed")
    print(f"Negatives: {n_ok}/{n_total} passed")
    overall = p_ok + n_ok
    total = p_total + n_total
    print(f"Overall: {overall}/{total} passed")

if __name__ == "__main__":
    main()
