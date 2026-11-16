#!/usr/bin/env python3
"""Decision-модель d1:2b (Liquid AI) на тех же 326 обращениях через /v1/systemone отдельной Ollama 0.40.15.
Вопрос типа choice: шесть категорий с теми же описаниями, что в промпте генеративных моделей.
python3 d1bench.py --url http://127.0.0.1:11435 --tag d1-2b [--limit N] [--repeat-ids 50]
Результат: scenarios/01b_class_big/results/<tag>/d1_2b.jsonl"""
import argparse, json, time, urllib.request, urllib.error, random
from pathlib import Path

D = Path("scenarios/01b_class_big")
ds = json.load(open(D / "dataset.json"))
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:11435"); ap.add_argument("--model", default="d1:2b")
ap.add_argument("--tag", default="d1-2b"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--repeat-ids", type=int, default=50, help="повторить N случайных обращений для проверки стабильности")
a = ap.parse_args()
out = D / "results" / a.tag; out.mkdir(parents=True, exist_ok=True)

QUESTION = {"category": {
    "type": "choice",
    "instructions": (f"Ты классификатор обращений в службу поддержки. Контекст: {ds['context']} "
                     "Отнеси обращение ровно к одной категории – по главной цели клиента."),
    "criteria": ds["categories"]}}


def ask(text):
    body = json.dumps({"model": a.model, "state": text, "questions": QUESTION, "keep_alive": "30m"}).encode()
    req = urllib.request.Request(a.url + "/v1/systemone", body, {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            res = json.loads(r.read())
    except urllib.error.HTTPError as e:
        res = {"answers": {"category": {}}, "error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"}
    return res, time.time() - t0


# прогрев посторонним текстом
ask("Здравствуйте, проверка связи.")
items = ds["items"][:a.limit] if a.limit else ds["items"]
with open(out / "d1_2b.jsonl", "w") as f:
    for it in items:
        res, w = ask(it["text"])
        ans = res["answers"]["category"]
        row = {"id": it["id"], "label": it["label"], "difficulty": it["difficulty"], "pred": ans.get("choice"),
               "ok": ans.get("choice") == it["label"], "probabilities": ans.get("probabilities"),
               "confidence": ans.get("confidence"), "wall_s": round(w, 3), "usage": res.get("usage"), "error": res.get("error"), "rep": 0}
        f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
        print(f"#{it['id']:>3} {str(row['pred']):>15} {'+' if row['ok'] else '-'} conf={row['confidence']} {w:.2f}s", flush=True)
    if a.repeat_ids:
        for it in random.Random(1).sample(items, min(a.repeat_ids, len(items))):
            res, w = ask(it["text"]); ans = res["answers"]["category"]
            f.write(json.dumps({"id": it["id"], "label": it["label"], "pred": ans.get("choice"), "ok": ans.get("choice") == it["label"],
                                "probabilities": ans.get("probabilities"), "confidence": ans.get("confidence"),
                                "wall_s": round(w, 3), "rep": 1}, ensure_ascii=False) + "\n")
print("D1BENCH_DONE")
