#!/usr/bin/env python3
"""Прогон тех же сценариев через любой OpenAI-совместимый API — «верхняя планка» для сравнения с локальными
моделями. Промпты, JSON-схемы и оценка берутся из scenario.py без изменений.

  API_BASE_URL=https://.../v1 API_KEY=... python3 apirunner.py scenarios/01b_class_big \
      --models deepseek/deepseek-v4.1-flash --tag api --think off [--ids 3,5] [--repeat 1] [--workers 4]

Адрес API: переменная API_BASE_URL. Ключ: переменная API_KEY или файл .env.api (API_KEY=...), в лог не пишется.
Если провайдер возвращает стоимость в usage.cost_rub, она копится в api_spend.json; при достижении API_BUDGET_RUB (по умолчанию 500) прогон останавливается."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner  # noqa: E402

BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
BUDGET = float(os.environ.get("API_BUDGET_RUB", "500"))
SPEND = HERE / "api_spend.json"
_lock = threading.Lock()
FORCED = set()   # модели, у которых рассуждение нельзя выключить: для них «off» = effort low


class BudgetExceeded(RuntimeError):
    pass


def _key():
    if os.environ.get("API_KEY"):
        return os.environ["API_KEY"]
    env = HERE / ".env.api"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("нет API_KEY: задайте переменную окружения или строку API_KEY=... в .env.api")


KEY = _key()


def spent() -> float:
    return json.loads(SPEND.read_text()).get("total_rub", 0.0) if SPEND.exists() else 0.0


def add_spend(model: str, rub: float):
    with _lock:
        d = json.loads(SPEND.read_text()) if SPEND.exists() else {"total_rub": 0.0, "by_model": {}}
        d["total_rub"] = round(d["total_rub"] + rub, 6)
        d["by_model"][model] = round(d["by_model"].get(model, 0.0) + rub, 6)
        SPEND.write_text(json.dumps(d, ensure_ascii=False, indent=1))


def chat(model, messages, schema, opts, think):
    if spent() >= BUDGET:
        raise BudgetExceeded(f"бюджет {BUDGET} ₽ исчерпан")
    p = {"model": model, "messages": messages, "temperature": opts.get("temperature", 0),
         "seed": opts.get("seed", 42), "reasoning": {"enabled": bool(think)}}
    np = opts.get("num_predict") or 512
    p["max_tokens"] = max(np, 8000) if think else max(np * 2, 400)
    if not think and model in FORCED:
        p["reasoning"] = {"effort": "low"}
        p["max_tokens"] = max(p["max_tokens"], 3000)
    if schema is not None:
        p["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
    last = None
    for attempt in range(4):
        data = json.dumps(p).encode()
        req = urllib.request.Request(BASE_URL + "/chat/completions", data=data, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                body = json.loads(r.read())
            wall = time.time() - t0
            u = body.get("usage") or {}
            add_spend(model, float(u.get("cost_rub") or 0))
            msg = (body.get("choices") or [{}])[0].get("message") or {}
            return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning") or "",
                    "wall": wall, "prompt_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
                    "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    "cost_rub": u.get("cost_rub"), "served_by": body.get("model"), "error": None,
                    "reasoning_forced": model in FORCED and not think}
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code == 400 and "mandatory" in last.lower() and not think and p["reasoning"] != {"effort": "low"}:
                with _lock:
                    FORCED.add(model)
                p["reasoning"] = {"effort": "low"}
                p["max_tokens"] = max(p["max_tokens"], 3000)
                continue
            if e.code not in (429, 500, 502, 503, 504):
                break
        except Exception as e:  # noqa
            last = repr(e)[:200]
        time.sleep(5 * (attempt + 1))
    return {"content": "", "reasoning": "", "wall": 0, "prompt_tokens": None, "output_tokens": None,
            "reasoning_tokens": None, "cost_rub": 0, "served_by": None, "error": last, "reasoning_forced": False}


def run_model(model, mod, ds, out, think, ids, repeat, workers):
    opts = {**runner.BASE_OPTIONS, **getattr(mod, "OPTIONS", {})}
    schema = mod.schema(ds) if hasattr(mod, "schema") else None
    items = [it for it in ds["items"] if not ids or it["id"] in ids] * repeat

    def one(it):
        r = chat(model, mod.build_messages(it, ds), schema, opts, think)
        parsed = runner.parse_json(r["content"]) if schema is not None else None
        try:
            sc = mod.score(it, parsed, r["content"], ds)
        except Exception as e:  # noqa
            sc = {"ok": False, "score_error": repr(e)[:200]}
        row = {"model": model, "id": it["id"], "ok": bool(sc.get("ok")), "score": sc, "raw": r["content"],
               "parsed": parsed, "error": r["error"], "reasoning_chars": len(r["reasoning"]), "wall_s": round(r["wall"], 3),
               "prompt_tokens": r["prompt_tokens"], "output_tokens": r["output_tokens"],
               "reasoning_tokens": r["reasoning_tokens"], "cost_rub": r["cost_rub"], "served_by": r["served_by"], "reasoning_forced": r["reasoning_forced"]}
        print(f"  [{model}] #{it['id']:>3} ok={row['ok']!s:5} {row['wall_s']:6.2f}s {row['cost_rub'] or 0:.4f}₽ "
              f"{(r['error'] or r['content'])[:60]!r}", flush=True)
        return row

    rows, stop = [], None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, it) for it in items]
        for f in futs:
            try:
                rows.append(f.result())
            except BudgetExceeded as e:
                stop = str(e)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    walls = [r["wall_s"] for r in rows if not r["error"]]
    n = len(rows)
    summ = {"model": model, "think": "on" if think else "off", "n": n,
            "accuracy": round(sum(r["ok"] for r in rows) / n, 3) if n else None,
            "errors": sum(1 for r in rows if r["error"]),
            "invalid_json": sum(1 for r in rows if schema is not None and r["parsed"] is None),
            "latency_avg_s": round(statistics.mean(walls), 2) if walls else None,
            "latency_p50_s": round(runner.pct(walls, 50), 2) if walls else None,
            "latency_p95_s": round(runner.pct(walls, 95), 2) if walls else None,
            "avg_prompt_tokens": round(statistics.mean(r["prompt_tokens"] or 0 for r in rows)) if n else None,
            "avg_output_tokens": round(statistics.mean(r["output_tokens"] or 0 for r in rows)) if n else None,
            "avg_reasoning_tokens": round(statistics.mean(r["reasoning_tokens"] or 0 for r in rows)) if n else None,
            "cost_rub": round(sum(r["cost_rub"] or 0 for r in rows), 4),
            "cost_per_1000_rub": round(1000 * sum(r["cost_rub"] or 0 for r in rows) / n, 2) if n else None,
            "reasoning_forced": sum(1 for r in rows if r.get("reasoning_forced")),
            "stopped": stop}
    if hasattr(mod, "summarize") and rows:
        try:
            summ.update(mod.summarize(rows, ds))
        except Exception as e:  # noqa
            summ["summarize_error"] = repr(e)[:200]
    return summ, stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--models", required=True)
    ap.add_argument("--tag", default="api-off")
    ap.add_argument("--think", choices=["on", "off"], default="off")
    ap.add_argument("--ids", default="")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    d = Path(a.scenario).resolve()
    mod, ds = runner.load_scenario(d)
    outdir = d / "results" / a.tag
    outdir.mkdir(parents=True, exist_ok=True)
    ids = {int(x) for x in a.ids.split(",") if x}
    sp = outdir / "summary.json"
    allsum = json.loads(sp.read_text()) if sp.exists() else {}
    print(f"API: {BASE_URL}  сценарий: {ds.get('title')}  потрачено: {spent():.2f} / {BUDGET} ₽", flush=True)
    for m in a.models.split(","):
        print(f"== {m} think={a.think}", flush=True)
        s, stop = run_model(m, mod, ds, outdir / (m.replace("/", "_").replace(":", "_") + ".jsonl"), a.think == "on",
                            ids, a.repeat, a.workers)
        print("   ", json.dumps(s, ensure_ascii=False), flush=True)
        allsum[m] = s
        sp.write_text(json.dumps(allsum, ensure_ascii=False, indent=1))
        if stop:
            print("BUDGET_STOP", flush=True)
            sys.exit(3)
    print(f"DONE  потрачено всего: {spent():.2f} ₽", flush=True)


if __name__ == "__main__":
    main()
