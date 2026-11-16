#!/usr/bin/env python3
"""Анализ d1:2b на 326 обращениях: точность, интервалы, McNemar против генеративных моделей,
калибровка уверенности и доля автоматических решений при пороге."""
import json, math, glob, statistics, os
from pathlib import Path
R = Path("scenarios/01b_class_big/results")
rows = [json.loads(l) for l in open(R / "d1-2b" / "d1_2b.jsonl")]
main = [r for r in rows if r["rep"] == 0]
errs = [r for r in main if not r.get("probabilities")]
print("ошибок запроса:", len(errs), [r["id"] for r in errs]); rep = [r for r in rows if r["rep"] == 1]
n = len(main); k = sum(r["ok"] for r in main)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def mcnemar_exact(b, c):
    m = b + c
    if m == 0:
        return 1.0
    x = min(b, c)
    p = sum(math.comb(m, i) for i in range(0, x + 1)) / 2 ** m
    return min(1.0, 2 * p)


w = sorted(r["wall_s"] for r in main)
print(f"d1:2b: {k}/{n} = {100*k/n:.1f}% {wilson(k,n)}; время p50 {statistics.median(w):.2f} сек, p95 {w[int(.95*(n-1))]:.2f}")
byid = {r["id"]: r for r in main}
if rep:
    same = sum(byid[r["id"]]["pred"] == r["pred"] and abs((byid[r["id"]]["confidence"] or 0) - (r["confidence"] or 0)) < 1e-3 for r in rep)
    print(f"повтор {len(rep)} обращений: тот же ответ и уверенность в {same}")
cats = sorted({r["label"] for r in main}); f1 = []
for c in cats:
    tp = sum(r["pred"] == c and r["label"] == c for r in main); fp = sum(r["pred"] == c and r["label"] != c for r in main)
    fn = sum(r["pred"] != c and r["label"] == c for r in main)
    p = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
    f1.append(2 * p * rc / (p + rc) if p + rc else 0)
print("macro F1", round(sum(f1) / len(f1), 3))
for d in ("easy", "medium", "hard"):
    rr = [r for r in main if r["difficulty"] == d]
    print(f"  {d}: {sum(r['ok'] for r in rr)}/{len(rr)}")

print("\nMcNemar против генеративных (ram8-r2, тот же набор):")
for f in sorted(glob.glob(str(R / "ram8-r2" / "*.jsonl"))):
    g = {json.loads(l)["id"]: json.loads(l) for l in open(f)}
    b = sum(1 for i, r in byid.items() if r["ok"] and not g[i]["ok"])
    c = sum(1 for i, r in byid.items() if not r["ok"] and g[i]["ok"])
    gk = sum(1 for i in byid if g[i]["ok"])
    print(f"  {os.path.basename(f)[:-6][:40]:40s} {gk}/{n} ({100*gk/n:.1f}%)  d1 лучше в {b}, хуже в {c}, p={mcnemar_exact(b,c):.3f}")

conf = lambda r: r["confidence"] if r.get("confidence") is not None else pmax(r)
pmax = lambda r: max(r["probabilities"].values()) if r.get("probabilities") else 0.0
print("\nКалибровка по максимальной вероятности (бины):")
ece = 0
for lo, hi in [(0, .5), (.5, .7), (.7, .9), (.9, .97), (.97, 1.01)]:
    rr = [r for r in main if lo <= pmax(r) < hi]
    if rr:
        acc = sum(r["ok"] for r in rr) / len(rr); mp = statistics.mean(pmax(r) for r in rr)
        ece += len(rr) / n * abs(acc - mp)
        print(f"  [{lo:.2f},{hi:.2f}) n={len(rr):3d} средняя вероятность {mp:.3f} точность {acc:.3f}")
print(f"  ECE {ece:.3f}")
print("\nПорог по максимальной вероятности: доля автоматически, точность автоматических, сколько в ручную:")
for t in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
    auto = [r for r in main if pmax(r) >= t]
    if auto:
        print(f"  >={t:.2f}: авто {len(auto)}/{n} ({100*len(auto)/n:.0f}%), точность {100*sum(r['ok'] for r in auto)/len(auto):.1f}%, в ручную {n-len(auto)}")
print("\nПо полю confidence:")
for t in (0.5, 0.7, 0.8, 0.9):
    auto = [r for r in main if conf(r) >= t]
    if auto:
        print(f"  >={t:.2f}: авто {len(auto)}/{n} ({100*len(auto)/n:.0f}%), точность {100*sum(r['ok'] for r in auto)/len(auto):.1f}%")
hl = json.load(open("human50_labels.json"))
agree = sum(1 for i, v in hl.items() if int(i) in byid and byid[int(i)]["pred"] == v["cat"])
print(f"\nс ручной разметкой 50: совпало {agree}/{len(hl)}")
