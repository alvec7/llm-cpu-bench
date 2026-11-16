#!/usr/bin/env python3
"""Проверка: эталонные решения D1–D6 проходят собственную оценку (в песочнице Docker)."""
import json, sys
sys.path.insert(0, ".")
import runner
GOLD = {"D1_sql": lambda it: {"sql": it["gold_sql"]}, "D2_python": lambda it: {"code": it["gold"]},
        "D3_traceback": lambda it: {"fixed_code": it["gold"], "cause": "-"}, "D4_pandas": lambda it: {"code": it["gold_code"]},
        "D5_regex_shell": lambda it: ({"pattern": it["gold"]} if it["kind"] == "regex" else {"command": it["gold"]})}
bad = 0
for d, f in GOLD.items():
    mod, ds = runner.load_scenario(runner.HERE / "scenarios" / d)
    res = [mod.score(it, f(it), "", ds)["ok"] for it in ds["items"]]
    bad += res.count(False)
    print(d, f"{sum(res)}/{len(res)}", flush=True)
print("GOLD_OK" if bad == 0 else f"GOLD_FAIL {bad}")
