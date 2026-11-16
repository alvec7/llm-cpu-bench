"""Сценарий 1. Классификация обращений клиентов по 6 категориям (structured output, enum)."""
from collections import Counter, defaultdict

OPTIONS = {"num_predict": 32}


def build_messages(item, ds):
    cats = "\n".join(f"- {k}: {v}" for k, v in ds["categories"].items())
    system = (
        f"Ты классификатор обращений в службу поддержки. Контекст: {ds['context']}\n"
        f"Отнеси обращение ровно к одной категории — по главной цели клиента:\n{cats}\n"
        'Ответь только JSON вида {"category": "<код категории>"}.'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": item["text"]}]


def schema(ds):
    return {
        "type": "object",
        "properties": {"category": {"type": "string", "enum": list(ds["categories"])}},
        "required": ["category"],
    }


def score(item, parsed, raw, ds):
    pred = (parsed or {}).get("category") if isinstance(parsed, dict) else None
    return {"ok": pred == item["label"], "pred": pred, "label": item["label"], "difficulty": item["difficulty"]}


def summarize(rows, ds):
    labels = list(ds["categories"])
    f1s = []
    for c in labels:
        tp = sum(1 for r in rows if r["score"]["pred"] == c and r["score"]["label"] == c)
        fp = sum(1 for r in rows if r["score"]["pred"] == c and r["score"]["label"] != c)
        fn = sum(1 for r in rows if r["score"]["pred"] != c and r["score"]["label"] == c)
        p = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * p * rc / (p + rc) if p + rc else 0)
    by_diff = defaultdict(list)
    for r in rows:
        by_diff[r["score"]["difficulty"]].append(r["ok"])
    errors = Counter(f"{r['score']['label']}→{r['score']['pred']}" for r in rows if not r["ok"])
    return {
        "macro_f1": round(sum(f1s) / len(f1s), 3),
        "acc_by_difficulty": {k: round(sum(v) / len(v), 2) for k, v in sorted(by_diff.items())},
        "confusions": dict(errors.most_common()),
        "wrong_ids": [r["id"] for r in rows if not r["ok"]],
    }
