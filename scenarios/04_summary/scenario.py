"""Сценарий 4. Суммаризация: 5 ключевых пунктов. Оценка — покрытие ключевых фактов (по шаблонам)
и «честность чисел»: каждое число в резюме должно встречаться в исходном тексте."""
import re

OPTIONS = {"num_ctx": 6144, "num_predict": 450}


def build_messages(item, ds):
    system = (
        "Ты готовишь краткое резюме документа для руководителя. "
        f"Выдели {ds['points']} самых важных пунктов: главные цифры, проблемы, решения и сроки. "
        "Каждый пункт — одно короткое предложение на русском языке. Используй только факты из текста, "
        'ничего не добавляй от себя. Ответь только JSON: {"points": ["...", "..."]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": item["text"]}]


def schema(ds):
    return {"type": "object",
            "properties": {"points": {"type": "array", "items": {"type": "string"},
                                      "minItems": ds["points"], "maxItems": ds["points"]}},
            "required": ["points"]}


def _n(s):
    return s.lower().replace("ё", "е").replace(" ", " ")


def _nums(s):
    out = []
    for m in re.findall(r"\d+(?:[  ]\d{3})*(?:[.,]\d+)?", s):
        out.append(m.replace(" ", "").replace(" ", "").replace(".", ","))
    return out


def score(item, parsed, raw, ds):
    pts = (parsed or {}).get("points") if isinstance(parsed, dict) else None
    pts = [p for p in pts if isinstance(p, str)] if isinstance(pts, list) else []
    summ = _n(" ".join(pts))
    covered = [f["fact"] for f in item["facts"] if any(re.search(p, summ) for p in f["any"])]
    src_nums = set(_nums(_n(item["text"]))) | set(item.get("allowed_numbers", []))
    bad_nums = [x for x in _nums(summ) if x not in src_nums]
    letters = re.findall(r"[a-zа-я]", summ)
    cyr = sum(1 for c in letters if "а" <= c <= "я") / len(letters) if letters else 0
    cov = len(covered) / len(item["facts"])
    traps = [t["trap"] for t in item.get("traps", []) if any(re.search(p, summ) for p in t["any"])]
    return {"ok": cov >= 0.5 and not bad_nums and not traps, "coverage": round(cov, 3), "covered": covered,
            "missed": [f["fact"] for f in item["facts"] if f["fact"] not in covered],
            "bad_numbers": bad_nums, "traps": traps, "cyrillic_share": round(cyr, 2), "n_points": len(pts), "points": pts}


def summarize(rows, ds):
    n = len(rows)
    return {
        "coverage_avg": round(sum(r["score"]["coverage"] for r in rows) / n, 3),
        "facts_covered": f"{sum(len(r['score']['covered']) for r in rows)}/{sum(len(r['score']['covered']) + len(r['score']['missed']) for r in rows)}",
        "unsupported_numbers": sum(len(r["score"]["bad_numbers"]) for r in rows),
        "min_cyrillic_share": min(r["score"]["cyrillic_share"] for r in rows),
        "per_text": {r["id"]: r["score"]["coverage"] for r in rows},
        "trap_hits": sum(len(r["score"].get("traps", [])) for r in rows),
    }
