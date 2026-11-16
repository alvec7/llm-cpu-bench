#!/usr/bin/env python3
"""Сборка RAG: чанкинг базы знаний, эмбеддинги (Ollama /api/embed), поиск топ-K для каждого вопроса.
Пишет dataset.json для runner.py и build_stats.json (время индексации и поиска).
python3 rag_build.py [--embed bge-m3] [--k 4]"""
import argparse, json, math, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--embed", default="bge-m3"); ap.add_argument("--k", type=int, default=4)
ap.add_argument("--words", type=int, default=170); a = ap.parse_args()


def url():
    if os.environ.get("OLLAMA_URL"): return os.environ["OLLAMA_URL"]
    ip = subprocess.check_output(["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}",
                                  os.environ.get("OLLAMA_CONTAINER", "ollama-ollama-1")], text=True).split()[0]
    return f"http://{ip}:11434"


U = url()


def embed(texts):
    req = urllib.request.Request(U + "/api/embed", data=json.dumps({"model": a.embed, "input": texts, "keep_alive": "10m"}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["embeddings"]


def chunks_of(name, text, target):
    title = text.strip().split("\n", 1)[0].strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], []
    for p in paras:
        cur.append(p)
        if sum(len(x.split()) for x in cur) >= target:
            out.append("\n".join(cur)); cur = [cur[-1]] if len(cur[-1].split()) < target // 2 else []  # небольшое перекрытие
    if cur and (not out or "\n".join(cur) not in out[-1]): out.append("\n".join(cur))
    return [{"id": f"{name}#{i+1}", "doc": name, "text": f"[{title}]\n{c}"} for i, c in enumerate(out)]


chunks = []
for f in sorted((HERE / "kb").glob("*.txt")):
    chunks += chunks_of(f.stem, f.read_text(), a.words)
t0 = time.time(); vecs = []
for i in range(0, len(chunks), 8):
    vecs += embed([c["text"] for c in chunks[i:i + 8]])
t_index = time.time() - t0


def cos(x, y):
    return sum(p * q for p, q in zip(x, y)) / (math.sqrt(sum(p * p for p in x)) * math.sqrt(sum(q * q for q in y)))


qs = json.loads((HERE / "questions.json").read_text()); items = []; t_q = []
for q in qs:
    t1 = time.time(); qv = embed([q["q"]])[0]
    ranked = sorted(range(len(chunks)), key=lambda i: -cos(qv, vecs[i]))[: a.k]
    t_q.append(time.time() - t1)
    ctx = [chunks[i] for i in ranked]
    hit = None if not q["answerable"] else any(c["doc"] == q["source"] and any(re.search(p, c["text"].lower().replace("ё", "е")) for p in (q.get("any") or q.get("all"))) for c in ctx)
    items.append({**q, "context": [{"id": c["id"], "text": c["text"]} for c in ctx], "retrieval_hit": hit})
ds = {"scenario": "09_rag", "title": "Ответы по базе знаний (RAG)", "embed_model": a.embed, "k": a.k, "n_chunks": len(chunks), "items": items}
(HERE / "dataset.json").write_text(json.dumps(ds, ensure_ascii=False, indent=1))
stats = {"embed_model": a.embed, "n_chunks": len(chunks), "kb_words": sum(len(c["text"].split()) for c in chunks),
         "index_s": round(t_index, 1), "per_chunk_s": round(t_index / len(chunks), 2), "query_s_avg": round(sum(t_q) / len(t_q), 2),
         "retrieval_hits": f"{sum(1 for i in items if i['retrieval_hit'])}/{sum(1 for i in items if i['answerable'])}",
         "misses": [i["id"] for i in items if i["retrieval_hit"] is False]}
(HERE / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1)); print(json.dumps(stats, ensure_ascii=False))
