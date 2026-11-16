#!/usr/bin/env python3
"""
Универсальный раннер экспериментов с лёгкими LLM на Ollama (только stdlib).

  python3 runner.py <scenario_dir> [--models m1,m2] [--repeat 1] [--tag base]

Сценарий — папка с файлами:
  dataset.json   — данные и эталоны (items[] с id)
  scenario.py    — build_messages(item, ds) -> list[dict]
                   schema(ds) -> dict | None      (JSON-схема для structured output)
                   score(item, parsed, raw, ds) -> dict  (обязательно ключ "ok": bool)
                   summarize(rows, ds) -> dict     (необязательно: доп. метрики качества)
                   OPTIONS = {...}                 (необязательно: параметры Ollama)

Результаты: <scenario_dir>/results/<tag>/<model>.jsonl и summary.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = (HERE / "models.txt").read_text().split() if (HERE / "models.txt").exists() else []
BASE_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 2048}


def ollama_url() -> str:
    if os.environ.get("OLLAMA_URL"):
        return os.environ["OLLAMA_URL"].rstrip("/")
    c = os.environ.get("OLLAMA_CONTAINER", "ollama-ollama-1")
    try:
        ip = subprocess.check_output(
            ["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}", c],
            text=True).split()[0]
        return f"http://{ip}:11434"
    except Exception:
        return "http://127.0.0.1:11434"


URL = ollama_url()


def api(path: str, payload: dict | None = None, timeout: int = 900) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if payload is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


class MemSampler(threading.Thread):
    """Пиковое потребление RAM и swap хоста во время прогона."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = False
        self.peak_used = 0
        self.peak_swap = 0
        self.swap0 = self.vm()

    @staticmethod
    def vm():
        d = {}
        for ln in open("/proc/vmstat"):
            k, v = ln.split()
            if k in ("pswpin", "pswpout"):
                d[k] = int(v)
        return d

    @staticmethod
    def read():
        m = {}
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":")
            m[k] = int(v.split()[0]) * 1024
        return m["MemTotal"] - m["MemAvailable"], m["SwapTotal"] - m["SwapFree"]

    def run(self):
        while not self.stop:
            u, s = self.read()
            self.peak_used, self.peak_swap = max(self.peak_used, u), max(self.peak_swap, s)
            time.sleep(0.5)


def load_scenario(d: Path):
    spec = importlib.util.spec_from_file_location("scenario", d / "scenario.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ds = json.loads((d / "dataset.json").read_text())
    return mod, ds


def parse_json(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").split("\n", 1)[-1]
    try:
        return json.loads(t)
    except Exception:
        s, e = t.find("{"), t.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except Exception:
                return None
    return None


def pct(vals, p):
    if not vals:
        return None
    v = sorted(vals)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def run_model(model: str, mod, ds, out: Path, repeat: int, extra_opts: dict | None = None,
              think_mode: str = "off", unique_prefix: bool = False) -> dict:
    gb = 1024 ** 3
    show = api("/api/show", {"model": model})
    caps = show.get("capabilities") or []
    opts = {**BASE_OPTIONS, **getattr(mod, "OPTIONS", {}), **(extra_opts or {})}
    schema = mod.schema(ds) if hasattr(mod, "schema") else None
    # think_mode: off — выключить рассуждение там, где оно есть; on — включить; auto — не трогать
    think = None if think_mode == "auto" or "thinking" not in caps else (think_mode == "on")

    def chat(messages):
        p = {"model": model, "messages": messages, "stream": False, "options": opts, "keep_alive": "15m"}
        if schema is not None:
            p["format"] = schema
        if think is not None:
            p["think"] = think
        t0 = time.time()
        r = api("/api/chat", p)
        r["_wall"] = time.time() - t0
        return r

    # прогрев: загрузка модели в память (cold start меряем отдельно).
    # Короткий посторонний запрос, чтобы первый пример датасета не попал в кэш промпта.
    w = chat([{"role": "user", "content": "Ответь одним словом: готов?"}])
    cold = {"wall_s": round(w["_wall"], 2), "load_s": round(w.get("load_duration", 0) / 1e9, 2)}
    ps = next((m for m in api("/api/ps").get("models", []) if m["name"] == model), {})

    sampler = MemSampler()
    sampler.start()
    rows = []
    with open(out, "w") as f:
        for rep in range(repeat):
            for it in ds["items"]:
                try:
                    msgs = mod.build_messages(it, ds)
                    if unique_prefix:  # без кэша: уникальная метка в самом начале промпта
                        import uuid
                        msgs[0] = {**msgs[0], "content": f"Запрос {uuid.uuid4().hex[:12]}.\n" + msgs[0]["content"]}
                    r = chat(msgs)
                    raw = r.get("message", {}).get("content", "")
                    thinking = r.get("message", {}).get("thinking", "")
                    parsed = parse_json(raw) if schema is not None else None
                    sc = mod.score(it, parsed, raw, ds)
                    err = None
                except Exception as e:  # noqa
                    r, raw, parsed, sc, err, thinking = {"_wall": 0}, "", None, {"ok": False}, str(e), ""
                row = {
                    "model": model, "rep": rep, "id": it["id"], "ts": round(time.time(), 1), "ok": bool(sc.get("ok")), "score": sc,
                    "raw": raw, "parsed": parsed, "error": err, "thinking_chars": len(thinking or ""),
                    "wall_s": round(r["_wall"], 3),
                    "prompt_tokens": r.get("prompt_eval_count"),
                    "output_tokens": r.get("eval_count"),
                    "prompt_s": round(r.get("prompt_eval_duration", 0) / 1e9, 3),
                    "gen_s": round(r.get("eval_duration", 0) / 1e9, 3),
                }
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{model}] #{it['id']:>3} ok={row['ok']!s:5} {row['wall_s']:6.2f}s  {raw[:70]!r}",
                      flush=True)
    sampler.stop = True
    sampler.join()
    api("/api/generate", {"model": model, "keep_alive": 0})  # выгрузить

    walls = [r["wall_s"] for r in rows if not r["error"]]
    p_tps = [r["prompt_tokens"] / r["prompt_s"] for r in rows if r["prompt_s"]]
    g_tps = [r["output_tokens"] / r["gen_s"] for r in rows if r["gen_s"]]
    n = len(rows)
    summ = {
        "model": model,
        "n": n,
        "accuracy": round(sum(r["ok"] for r in rows) / n, 3) if n else None,
        "errors": sum(1 for r in rows if r["error"]),
        "invalid_json": sum(1 for r in rows if schema is not None and r["parsed"] is None),
        "latency_avg_s": round(statistics.mean(walls), 2) if walls else None,
        "latency_p50_s": round(pct(walls, 50), 2) if walls else None,
        "latency_p95_s": round(pct(walls, 95), 2) if walls else None,
        "prompt_tok_s": round(statistics.median(p_tps), 1) if p_tps else None,
        "gen_tok_s": round(statistics.median(g_tps), 1) if g_tps else None,
        "avg_prompt_tokens": round(statistics.mean(r["prompt_tokens"] or 0 for r in rows), 0),
        "avg_output_tokens": round(statistics.mean(r["output_tokens"] or 0 for r in rows), 1),
        "items_per_hour_1stream": round(3600 / statistics.mean(walls)) if walls else None,
        "cold_start": cold,
        "model_ram_gb": round((ps.get("size") or 0) / gb, 2),
        "host_peak_used_gb": round(sampler.peak_used / gb, 2),
        "host_peak_swap_gb": round(sampler.peak_swap / gb, 2),
        "swap_io_mb": round(sum(sampler.vm()[k] - sampler.swap0[k] for k in sampler.swap0) * 4 / 1024, 1),
        "think": {None: "n/a", False: "off", True: "on"}[think],
        "unique_prefix": unique_prefix,
        "options": opts,
        "quant": (show.get("details") or {}).get("quantization_level"),
        "params": (show.get("details") or {}).get("parameter_size"),
    }
    if hasattr(mod, "summarize"):
        summ.update(mod.summarize(rows, ds))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--opt", action="append", default=[], help="параметр Ollama key=value, напр. num_predict=1024")
    ap.add_argument("--think", choices=["off", "on", "auto"], default="off")
    ap.add_argument("--unique-prefix", action="store_true", help="уникальная метка в начале промпта (кэш не работает)")
    ap.add_argument("--sample", type=int, default=0, help="случайная подвыборка N примеров (seed 1)")
    ap.add_argument("--ids", default="", help="только эти id через запятую (перегон отдельных заданий)")
    a = ap.parse_args()
    d = Path(a.scenario).resolve()
    mod, ds = load_scenario(d)
    if a.ids:
        want = set(a.ids.split(","))
        ds = {**ds, "items": [it for it in ds["items"] if str(it["id"]) in want]}
    if a.sample:
        import random
        ids = sorted(random.Random(1).sample(range(len(ds["items"])), a.sample))
        ds = {**ds, "items": [ds["items"][i] for i in ids]}
    outdir = d / "results" / a.tag
    outdir.mkdir(parents=True, exist_ok=True)
    models = [m for m in a.models.split(",") if m]
    print(f"Ollama: {URL}  сценарий: {ds.get('title')}  моделей: {len(models)}", flush=True)
    summary_path = outdir / "summary.json"
    allsum = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for m in models:
        print(f"== {m}", flush=True)
        try:
            extra = {k: json.loads(v) for k, v in (o.split("=", 1) for o in a.opt)}
            s = run_model(m, mod, ds, outdir / (m.replace("/", "_").replace(":", "_") + ".jsonl"), a.repeat,
                          extra, a.think, a.unique_prefix)
        except Exception as e:  # noqa
            s = {"model": m, "fatal": str(e)}
        print("   ", json.dumps(s, ensure_ascii=False), flush=True)
        allsum[m] = s
        summary_path.write_text(json.dumps(allsum, ensure_ascii=False, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
