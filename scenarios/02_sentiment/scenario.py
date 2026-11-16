"""Сценарий 2. Тональность отзывов: sentiment (enum) + aspects (список enum) + main_reason (текст)."""
from collections import Counter, defaultdict

OPTIONS = {"num_predict": 128}


def build_messages(item, ds):
    s = "\n".join(f"- {k}: {v}" for k, v in ds["sentiments"].items())
    a = "\n".join(f"- {k}: {v}" for k, v in ds["aspects"].items())
    system = (
        f"Ты анализируешь отзывы покупателей. Контекст: {ds['context']}\n"
        f"Определи общую тональность отзыва (sentiment):\n{s}\n"
        f"Перечисли, о чём говорится в отзыве (aspects), одно или несколько значений:\n{a}\n"
        "В main_reason одной короткой фразой по-русски напиши главную причину оценки клиента.\n"
        'Ответь только JSON: {"sentiment": "...", "aspects": ["..."], "main_reason": "..."}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": item["text"]}]


def schema(ds):
    return {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": list(ds["sentiments"])},
            "aspects": {"type": "array", "items": {"type": "string", "enum": list(ds["aspects"])}, "minItems": 1},
            "main_reason": {"type": "string"},
        },
        "required": ["sentiment", "aspects", "main_reason"],
    }


def score(item, parsed, raw, ds):
    p = parsed if isinstance(parsed, dict) else {}
    pred = p.get("sentiment")
    pa = set(p.get("aspects") or []) if isinstance(p.get("aspects"), list) else set()
    ga = set(item["aspects"])
    tp = len(pa & ga)
    prec = tp / len(pa) if pa else 0
    rec = tp / len(ga) if ga else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return {"ok": pred == item["label"], "pred": pred, "label": item["label"], "group": item["group"],
            "aspects_pred": sorted(pa), "aspects_f1": round(f1, 3), "reason": p.get("main_reason")}


def summarize(rows, ds):
    labels = list(ds["sentiments"])
    f1s = []
    for c in labels:
        tp = sum(1 for r in rows if r["score"]["pred"] == c and r["score"]["label"] == c)
        fp = sum(1 for r in rows if r["score"]["pred"] == c and r["score"]["label"] != c)
        fn = sum(1 for r in rows if r["score"]["pred"] != c and r["score"]["label"] == c)
        p = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * p * rc / (p + rc) if p + rc else 0)
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["score"]["group"]].append(r["ok"])
    return {
        "macro_f1": round(sum(f1s) / len(f1s), 3),
        "aspects_f1": round(sum(r["score"]["aspects_f1"] for r in rows) / len(rows), 3),
        "acc_by_group": {k: f"{sum(v)}/{len(v)}" for k, v in by_group.items()},
        "confusions": dict(Counter(f"{r['score']['label']}→{r['score']['pred']}" for r in rows if not r["ok"]).most_common()),
        "wrong_ids": [r["id"] for r in rows if not r["ok"]],
    }
