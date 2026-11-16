"""Сценарий 7. Объявления для Яндекс Директа: 5 вариантов (заголовок ≤56, текст ≤81 символа)."""
import difflib
import re

OPTIONS = {"num_predict": 600, "temperature": 0.7}  # креативная задача: немного разнообразия, seed фиксирован
BANNED = r"лучш|№\s?1\b|номер один|самы[йеахм]\w*|гарантированн|дешевле всех|первы[йе] в росс|уникальн"


def build_messages(item, ds):
    L = ds["limits"]
    system = (
        "Ты опытный специалист по контекстной рекламе в Яндекс Директе. По брифу напиши "
        f"{ds['n_ads']} разных вариантов текстового объявления на русском языке.\n"
        f"Требования: заголовок — не длиннее {L['title']} символов, текст — не длиннее {L['text']} символов (с пробелами и знаками). "
        f"В каждом заголовке используй ключевую фразу «{item['keyword']}» (можно в другой форме слова). "
        "Используй только факты и цифры из брифа, ничего не придумывай. Не используй превосходную степень "
        "(«лучший», «самый», «№1») — это запрещено законом о рекламе. Варианты должны отличаться подходом: выгода, цена, удобство, призыв.\n"
        'Ответь только JSON: {"ads": [{"title": "...", "text": "..."}, ...]}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": f"Бриф:\n{item['brief']}"}]


def schema(ds):
    ad = {"type": "object", "properties": {"title": {"type": "string"}, "text": {"type": "string"}}, "required": ["title", "text"]}
    return {"type": "object", "properties": {"ads": {"type": "array", "items": ad, "minItems": ds["n_ads"], "maxItems": ds["n_ads"]}},
            "required": ["ads"]}


def _nums(s):
    return [m.replace(" ", "").replace(" ", "") for m in re.findall(r"\d+(?:[  ]\d{3})*(?:[.,]\d+)?", s)]


def _ad_nums(s):
    """Числа из объявления; «15 тыс.», «15к», «250 тысяч» приводим к 15000."""
    out = []
    for m in re.finditer(r"(\d+(?:[ \u00a0]\d{3})*(?:[.,]\d+)?)\s?(тыс|к\b|млн)?", s):
        n = m.group(1).replace(" ", "").replace("\u00a0", "")
        if m.group(2) and re.fullmatch(r"\d+", n):
            n = str(int(n) * (1_000_000 if m.group(2) == "млн" else 1000))
        out.append(n)
    return out


def check_ad(ad, item, ds):
    t, x = str(ad.get("title", "")).strip(), str(ad.get("text", "")).strip()
    tl = (t + " " + x).lower().replace("ё", "е")
    brief = item["brief"].lower().replace("ё", "е")
    brief_nums = set(_nums(brief))
    bad_nums = [n for n in _ad_nums((t + " " + x).lower()) if n not in brief_nums]
    banned = re.findall(BANNED, tl)
    if re.search(r"бесплатн", tl) and not re.search(r"бесплатн", brief):
        banned.append("бесплатн (нет в брифе)")
    c = {"title_len": len(t), "text_len": len(x), "title_ok": 0 < len(t) <= ds["limits"]["title"],
         "text_ok": 0 < len(x) <= ds["limits"]["text"], "keyword": bool(re.search(item["kw_re"], t.lower().replace("ё", "е"))),
         "bad_nums": bad_nums, "banned": banned}
    c["usable"] = c["title_ok"] and c["text_ok"] and c["keyword"] and not bad_nums and not banned
    return c


def score(item, parsed, raw, ds):
    ads = (parsed or {}).get("ads") if isinstance(parsed, dict) else None
    ads = [a for a in ads if isinstance(a, dict)] if isinstance(ads, list) else []
    checks = [check_ad(a, item, ds) for a in ads]
    titles = [str(a.get("title", "")).lower().strip() for a in ads]
    dup = sum(1 for i in range(len(titles)) for j in range(i + 1, len(titles))
              if difflib.SequenceMatcher(None, titles[i], titles[j]).ratio() > 0.85)
    usable = sum(c["usable"] for c in checks)
    return {"ok": usable >= 3, "n_ads": len(ads), "usable": usable, "near_duplicates": dup, "checks": checks,
            "ads": ads}


def summarize(rows, ds):
    allc = [c for r in rows for c in r["score"].get("checks", [])]
    n = len(rows) * ds["n_ads"]
    f = lambda k: sum(1 for c in allc if c[k])
    return {
        "usable_ads": f"{sum(r['score'].get('usable', 0) for r in rows)}/{n}",
        "title_within": f"{f('title_ok')}/{n}", "text_within": f"{f('text_ok')}/{n}", "keyword_in_title": f"{f('keyword')}/{n}",
        "ads_with_bad_numbers": sum(1 for c in allc if c["bad_nums"]), "ads_with_banned": sum(1 for c in allc if c["banned"]),
        "near_duplicate_pairs": sum(r["score"].get("near_duplicates", 0) for r in rows),
        "avg_title_len": round(sum(c["title_len"] for c in allc) / len(allc), 1) if allc else None,
        "avg_text_len": round(sum(c["text_len"] for c in allc) / len(allc), 1) if allc else None,
    }
