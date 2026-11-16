"""Сценарий 3. Извлечение полей из заявки в JSON. Оценка по каждому полю с нормализацией."""
import re
from collections import Counter

OPTIONS = {"num_predict": 160}
LEGAL = r"\b(ооо|оао|зао|пао|ао|мбоу|гбу|llc|ltd)\b"


def build_messages(item, ds):
    rules = "\n".join(f"- {r}" for r in ds["rules"])
    system = (
        f"Ты извлекаешь данные из входящих заявок. Контекст: {ds['context']}\n"
        f"Правила:\n{rules}\n"
        "Ответь только JSON с полями contact_name, company, phone, email, budget_rub, deadline."
    )
    user = f"Дата письма: {item['date']}\n\n{item['text']}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def schema(ds):
    s = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {"contact_name": s, "company": s, "phone": s, "email": s,
                       "budget_rub": {"type": ["integer", "null"]}, "deadline": s},
        "required": ds["fields"],
    }


def _toks(x):
    return set(re.findall(r"[a-zа-яё0-9]+", str(x).lower().replace("ё", "е")))


def norm(field, v):
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none", "нет")):
        return None
    if field == "phone":
        d = re.sub(r"\D", "", str(v))
        return ("7" + d[-10:]) if len(d) >= 10 else d
    if field == "budget_rub":
        try:
            return int(float(str(v).replace(" ", "").replace(",", ".")))
        except ValueError:
            return str(v)
    if field == "email":
        return str(v).strip().lower()
    if field == "company":
        return _toks(re.sub(LEGAL, " ", str(v).lower()))
    if field == "contact_name":
        return _toks(v)
    return str(v).strip()


def field_ok(field, pred, gold):
    p, g = norm(field, pred), norm(field, gold)
    if g is None or p is None:
        return p is None and g is None
    if field in ("contact_name", "company"):
        return g <= p  # все слова эталона есть в ответе (отчество допустимо)
    return p == g


def score(item, parsed, raw, ds):
    p = parsed if isinstance(parsed, dict) else {}
    res = {f: field_ok(f, p.get(f), item["gold"][f]) for f in ds["fields"]}
    halluc = [f for f in ds["fields"] if item["gold"][f] is None and norm(f, p.get(f)) is not None]
    missed = [f for f in ds["fields"] if item["gold"][f] is not None and norm(f, p.get(f)) is None]
    return {"ok": all(res.values()), "fields": res, "n_ok": sum(res.values()), "halluc": halluc, "missed": missed,
            "pred": {f: p.get(f) for f in ds["fields"]}, "difficulty": item["difficulty"]}


def summarize(rows, ds):
    per = Counter()
    for r in rows:
        for f, ok in r["score"]["fields"].items():
            per[f] += ok
    n = len(rows)
    return {
        "field_accuracy": round(sum(r["score"]["n_ok"] for r in rows) / (n * len(ds["fields"])), 3),
        "by_field": {f: f"{per[f]}/{n}" for f in ds["fields"]},
        "hallucinated_fields": sum(len(r["score"]["halluc"]) for r in rows),
        "missed_fields": sum(len(r["score"]["missed"]) for r in rows),
        "wrong": {r["id"]: [f for f, ok in r["score"]["fields"].items() if not ok] for r in rows if not r["ok"]},
    }
