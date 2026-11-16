#!/usr/bin/env python3
"""Нагрузочный тест: N запросов с заданной параллельностью, профили из готовых сценариев.
python3 loadtest.py --model qwen3.5:4b --profile short|medium --conc 1,2,4 --n 12 --tag A
Результат дописывается в loadtest/results.jsonl"""
import argparse, json, statistics, sys, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner  # noqa: E402  (api, MemSampler, load_scenario, BASE_OPTIONS)

PROFILES = {"short": "scenarios/01_classification", "medium": "scenarios/09_rag"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--profile", choices=PROFILES, required=True)
ap.add_argument("--conc", default="1,2,4"); ap.add_argument("--n", type=int, default=12); ap.add_argument("--tag", default="A")
a = ap.parse_args()

mod, ds = runner.load_scenario(HERE / PROFILES[a.profile])
opts = {**runner.BASE_OPTIONS, **getattr(mod, "OPTIONS", {})}
schema = mod.schema(ds)
caps = runner.api("/api/show", {"model": a.model}).get("capabilities") or []
items = ds["items"]


def call(i):
    it = items[i % len(items)]
    msgs = mod.build_messages(it, ds)
    # уникальная метка в начале сообщения пользователя: общий system-промпт может кэшироваться (как в жизни),
    # а сам запрос — нет (иначе llama.cpp переиспользует кэш и меряет нереально быстро)
    msgs[-1] = {**msgs[-1], "content": f"[запрос {uuid.uuid4().hex[:8]}]\n" + msgs[-1]["content"]}
    p = {"model": a.model, "messages": msgs, "stream": False, "options": opts,
         "format": schema, "keep_alive": "15m"}
    if "thinking" in caps:
        p["think"] = False
    t0 = time.time()
    try:
        r = runner.api("/api/chat", p, timeout=1200)
        return {"ok": True, "wall": time.time() - t0, "pt": r.get("prompt_eval_count"), "ps": r.get("prompt_eval_duration", 0) / 1e9,
                "gt": r.get("eval_count"), "gs": r.get("eval_duration", 0) / 1e9, "load": r.get("load_duration", 0) / 1e9}
    except Exception as e:  # noqa
        return {"ok": False, "wall": time.time() - t0, "err": str(e)[:200]}


# прогрев: загрузка модели с нужным num_ctx
w = call(0)
print(f"warmup {a.model} {a.profile}: {w['wall']:.1f}s ok={w['ok']}", flush=True)
out = HERE / "loadtest"; out.mkdir(exist_ok=True)
for c in [int(x) for x in a.conc.split(",")]:
    s = runner.MemSampler(); s.start()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=c) as ex:
        res = list(ex.map(call, range(1, a.n + 1)))
    total = time.time() - t0
    s.stop = True; s.join()
    good = [r for r in res if r["ok"]]
    walls = sorted(r["wall"] for r in good)
    pct = lambda q: round(walls[min(len(walls) - 1, int(round(q * (len(walls) - 1))))], 1) if walls else None
    ps = runner.api("/api/ps").get("models", [])
    row = {"tag": a.tag, "model": a.model, "profile": a.profile, "conc": c, "n": a.n, "ok": len(good), "errors": a.n - len(good),
           "total_s": round(total, 1), "req_per_hour": round(len(good) / total * 3600) if total else None,
           "lat_p50": pct(0.5), "lat_p95": pct(0.95), "lat_max": round(max(walls), 1) if walls else None,
           "gen_tok_s_per_req": round(statistics.median([r["gt"] / r["gs"] for r in good if r.get("gs")]), 1) if good else None,
           "prompt_tok_s_per_req": round(statistics.median([r["pt"] / r["ps"] for r in good if r.get("ps")]), 1) if good else None,
           "reloads": sum(1 for r in good if r.get("load", 0) > 1),
           "peak_used_gb": round(s.peak_used / 1024 ** 3, 2), "swap_io_mb": round(sum(s.vm()[k] - s.swap0[k] for k in s.swap0) * 4 / 1024, 1),
           "model_mem_gb": round(sum(m.get("size", 0) for m in ps) / 1024 ** 3, 2),
           "errs": [r["err"] for r in res if not r["ok"]][:3]}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
runner.api("/api/generate", {"model": a.model, "keep_alive": 0})
