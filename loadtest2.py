#!/usr/bin/env python3
"""Нагрузочный тест v2 (закрытая модель нагрузки): C клиентов шлют запросы друг за другом, без пауз.
Уровень длится, пока не выполнены оба условия: прошло --min-s секунд и завершено --min-n запросов.
Новые запросы после этого не отправляются, ждем завершения начатых.
Режим кэша: shared — уникальная метка в сообщении пользователя (общий системный промпт может браться из кэша);
nocache — уникальная метка в самом начале системного промпта (кэш префикса не срабатывает).
Ответы стримятся, чтобы мерить время до первого токена (TTFT).
python3 loadtest2.py --model M --profile short|long --conc 1,2,4 --cache shared,nocache --min-s 300 --min-n 40 --rep 2
Сводка -> loadtest2/results.jsonl, все запросы -> loadtest2/requests.jsonl"""
import argparse, json, statistics, sys, threading, time, urllib.request, uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner  # noqa: E402

PROFILES = {"short": "scenarios/01b_class_big", "long": "scenarios/09_rag"}
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--profile", choices=PROFILES, required=True)
ap.add_argument("--conc", default="1,2,4"); ap.add_argument("--cache", default="shared")
ap.add_argument("--min-s", type=float, default=300); ap.add_argument("--min-n", type=int, default=40)
ap.add_argument("--rep", type=int, default=1); ap.add_argument("--tag", default="v2")
a = ap.parse_args()

mod, ds = runner.load_scenario(HERE / PROFILES[a.profile])
opts = {**runner.BASE_OPTIONS, **getattr(mod, "OPTIONS", {})}
schema = mod.schema(ds)
caps = runner.api("/api/show", {"model": a.model}).get("capabilities") or []
items = ds["items"]
OUT = HERE / "loadtest2"; OUT.mkdir(exist_ok=True)
lock = threading.Lock()
counter = {"i": 0}


def call(mode):
    with lock:
        i = counter["i"]; counter["i"] += 1
    it = items[i % len(items)]
    msgs = [dict(m) for m in mod.build_messages(it, ds)]
    tag = uuid.uuid4().hex[:12]
    if mode == "nocache":
        msgs[0]["content"] = f"Запрос {tag}.\n" + msgs[0]["content"]
    else:
        msgs[-1]["content"] = f"[запрос {tag}]\n" + msgs[-1]["content"]
    p = {"model": a.model, "messages": msgs, "stream": True, "options": opts, "format": schema, "keep_alive": "15m"}
    if "thinking" in caps:
        p["think"] = False
    req = urllib.request.Request(runner.URL + "/api/chat", data=json.dumps(p).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""; last = {}
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                ch = json.loads(line)
                c = (ch.get("message") or {}).get("content", "")
                if c and ttft is None:
                    ttft = time.time() - t0
                text += c
                if ch.get("done"):
                    last = ch; break
        parsed = runner.parse_json(text)
        sc = mod.score(it, parsed, text, ds)
        return {"ok": True, "t0": round(t0, 2), "wall": round(time.time() - t0, 3), "ttft": round(ttft or 0, 3),
                "pt": last.get("prompt_eval_count"), "ps": round(last.get("prompt_eval_duration", 0) / 1e9, 3),
                "gt": last.get("eval_count"), "gs": round(last.get("eval_duration", 0) / 1e9, 3),
                "load": round(last.get("load_duration", 0) / 1e9, 3), "correct": bool(sc.get("ok")), "item": it["id"]}
    except Exception as e:  # noqa
        return {"ok": False, "t0": round(t0, 2), "wall": round(time.time() - t0, 3), "err": str(e)[:200], "item": it["id"]}


def level(c, mode, rep):
    res = []; start = time.time(); stop = threading.Event()

    def worker():
        while not stop.is_set():
            r = call(mode)
            with lock:
                res.append(r)
                if time.time() - start >= a.min_s and len(res) >= a.min_n:
                    stop.set()

    s = runner.MemSampler(); s.start()
    th = [threading.Thread(target=worker) for _ in range(c)]
    for t in th: t.start()
    for t in th: t.join()
    total = time.time() - start
    s.stop = True; s.join()
    good = [r for r in res if r["ok"]]
    def pct(vals, q):
        v = sorted(vals)
        return round(v[min(len(v) - 1, int(round(q * (len(v) - 1))))], 2) if v else None
    walls = [r["wall"] for r in good]; ttfts = [r["ttft"] for r in good]
    row = {"tag": a.tag, "model": a.model, "profile": a.profile, "cache": mode, "conc": c, "rep": rep,
           "n": len(res), "ok": len(good), "errors": len(res) - len(good), "duration_s": round(total, 1),
           "req_per_hour": round(len(good) / total * 3600) if total else None,
           "lat_p50": pct(walls, .5), "lat_p95": pct(walls, .95), "lat_p99": pct(walls, .99), "lat_max": pct(walls, 1),
           "ttft_p50": pct(ttfts, .5), "ttft_p95": pct(ttfts, .95),
           "in_tok_med": statistics.median([r["pt"] for r in good if r.get("pt")]) if good else None,
           "out_tok_med": statistics.median([r["gt"] for r in good if r.get("gt")]) if good else None,
           "gen_tok_s_med": round(statistics.median([r["gt"] / r["gs"] for r in good if r.get("gs")]), 2) if good else None,
           "accuracy": round(sum(r["correct"] for r in good) / len(good), 3) if good else None,
           "reloads": sum(1 for r in good if r.get("load", 0) > 1),
           "peak_used_gb": round(s.peak_used / 1024 ** 3, 2),
           "swap_io_mb": round(sum(s.vm()[k] - s.swap0[k] for k in s.swap0) * 4 / 1024, 1),
           "errs": [r["err"] for r in res if not r["ok"]][:3], "start_ts": round(start, 1)}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(OUT / "requests.jsonl", "a") as f:
        for r in res:
            f.write(json.dumps({"tag": a.tag, "model": a.model, "profile": a.profile, "cache": mode, "conc": c,
                                "rep": rep, **r}, ensure_ascii=False) + "\n")


w = call("shared")  # прогрев: загрузка модели с нужным num_ctx
print(f"warmup {a.model} {a.profile}: {w['wall']:.1f}s ok={w['ok']}", flush=True)
for rep in range(a.rep):
    for mode in a.cache.split(","):
        for c in [int(x) for x in a.conc.split(",")]:
            level(c, mode, rep)
runner.api("/api/generate", {"model": a.model, "keep_alive": 0})
