#!/usr/bin/env python3
"""Сводная таблица по сценарию: python3 report.py scenarios/<NN> [--tag base]"""
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "base"
s = json.loads((d / "results" / tag / "summary.json").read_text())
cols = ["accuracy", "macro_f1", "latency_avg_s", "latency_p95_s", "prompt_tok_s", "gen_tok_s",
        "items_per_hour_1stream", "model_ram_gb", "swap_io_mb"]
print("| model | " + " | ".join(cols) + " | cold_s | wrong_ids |")
print("|" + "---|" * (len(cols) + 3))
for m, v in sorted(s.items(), key=lambda kv: -(kv[1].get("accuracy") or 0)):
    if "fatal" in v:
        print(f"| {m} | FATAL {v['fatal'][:80]} |"); continue
    print(f"| {m} | " + " | ".join(str(v.get(c, "")) for c in cols) +
          f" | {v['cold_start']['wall_s']} | {v.get('wrong_ids')} |")
