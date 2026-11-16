#!/usr/bin/env python3
"""Пересчитать оценки по сохранённым ответам после правки score() — без повторного прогона моделей.
python3 rescore.py scenarios/<NN> [--tag base]"""
import importlib.util, json, sys
from pathlib import Path
d = Path(sys.argv[1]); tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "base"
spec = importlib.util.spec_from_file_location("scenario", d / "scenario.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ds = json.loads((d / "dataset.json").read_text()); items = {i["id"]: i for i in ds["items"]}
out = d / "results" / tag; summ = json.loads((out / "summary.json").read_text())
for f in out.glob("*.jsonl"):
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    for r in rows:
        r["score"] = mod.score(items[r["id"]], r["parsed"], r["raw"], ds); r["ok"] = bool(r["score"]["ok"])
    f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    m = rows[0]["model"]
    summ[m]["accuracy"] = round(sum(r["ok"] for r in rows) / len(rows), 3)
    if hasattr(mod, "summarize"):
        summ[m].update(mod.summarize(rows, ds))
    summ[m]["rescored"] = True
(out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
print("rescored", len(summ), "models")
