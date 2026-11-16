"""Сценарий 8. Нормализация пачки из 10 строк (телефоны / компании / адреса)."""
import re

OPTIONS = {"num_predict": 700}
N = 10


def build_messages(item, ds):
    rows = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(item["rows"]))
    system = ("Ты нормализуешь данные в CRM. " + item["rules"] +
              f" Верни ровно {N} значений в том же порядке, что и входные строки. "
              'Ответь только JSON: {"rows": ["...", null, ...]}')
    return [{"role": "system", "content": system}, {"role": "user", "content": rows}]


def schema(ds):
    return {"type": "object", "properties": {"rows": {"type": "array", "items": {"type": ["string", "null"]},
                                                      "minItems": N, "maxItems": N}}, "required": ["rows"]}


def _n(s):
    return re.sub(r"\s+", " ", str(s).lower().replace("ё", "е")).strip()


def _company(s):
    s = _n(s)
    m = re.match(r"^(ооо|ао|пао|зао|ип)\s+(.*)$", s)
    form, name = (m.group(1), m.group(2)) if m else ("", s)
    return form, re.sub(r"[«»\"'“”„]", "", name).strip()


TYPES = [(r"ул\.?|улица", "ул"), (r"пр-т|пр-кт|проспект|пр\.?", "пр-т"), (r"пер\.?|переулок", "пер"),
         (r"б-р|бульвар", "б-р"), (r"ш\.?|шоссе", "ш"), (r"наб\.?|набережная", "наб")]


CITIES = {"москва", "санкт-петербург", "казань", "екатеринбург", "нижний новгород", "новосибирск",
          "мск", "спб", "ебург"}  # «мск» и т.п. оставлены, чтобы не засчитывать их как улицу; при сравнении не совпадут с эталоном


def _street(seg):
    for pat, canon in TYPES:
        m = re.match(rf"^(?:{pat})\s+(.+)$", seg) or re.match(rf"^(.+?)\s+(?:{pat})$", seg)
        if m:
            return (canon, m.group(1).strip())
    return None


def _addr(s):
    """Разбор адреса на компоненты — снисходительно к оформлению (г./дом/д., порядок типа улицы).
    Строгое соответствие формату считается отдельно (strict)."""
    out = {"city": None, "street": None, "house": None, "flat": None, "str": None}
    for i, seg in enumerate(x.strip() for x in _n(s).split(",")):
        if not seg:
            continue
        if re.match(r"^г\.?\s+", seg):
            out["city"] = re.sub(r"^г\.?\s+", "", seg)
        elif m := re.match(r"^(?:д\.?|дом)\s*(\w+)$", seg):
            out["house"] = m.group(1)
        elif m := re.match(r"^(кв|оф)\.?\s*(\w+)$", seg):
            out["flat"] = (m.group(1), m.group(2))
        elif m := re.match(r"^стр\.?\s*(\w+)$", seg):
            out["str"] = m.group(1)
        elif re.match(r"^\d+\w*$", seg) and out["house"] is None:
            out["house"] = seg
        elif st := _street(seg):
            out["street"] = st
        elif out["city"] is None and seg in CITIES:
            out["city"] = seg
        elif out["street"] is None and not re.search(r"\d", seg):
            out["street"] = ("ул", seg)  # тип не указан → по правилу «ул.»
        else:
            out["junk"] = seg
    return out


def row_ok(kind, pred, gold):
    if gold is None or pred is None or (isinstance(pred, str) and _n(pred) in ("", "null", "none", "нет")):
        p_null = pred is None or _n(pred) in ("", "null", "none", "нет")
        return gold is None and p_null
    if kind == "phone":
        return _n(pred) == _n(gold)
    if kind == "company":
        return _company(pred) == _company(gold)
    return _addr(pred) == _addr(gold)


def score(item, parsed, raw, ds):
    rows = (parsed or {}).get("rows") if isinstance(parsed, dict) else None
    rows = rows if isinstance(rows, list) else []
    res = []
    for i, g in enumerate(item["gold"]):
        p = rows[i] if i < len(rows) else None
        res.append({"in": item["rows"][i], "pred": p, "gold": g, "ok": row_ok(item["kind"], p, g),
                    "strict": (p == g) or (p is None and g is None)})
    n_ok = sum(r["ok"] for r in res)
    return {"ok": n_ok >= 8, "kind": item["kind"], "n_ok": n_ok, "n_strict": sum(r["strict"] for r in res),
            "n_returned": len(rows), "rows": res}


def summarize(rows, ds):
    tot = len(rows) * N
    return {
        "rows_correct": f"{sum(r['score'].get('n_ok', 0) for r in rows)}/{tot}",
        "rows_strict": f"{sum(r['score'].get('n_strict', 0) for r in rows)}/{tot}",
        "by_kind": {r["score"].get("kind", r["id"]): r["score"].get("n_ok") for r in rows},
        "wrong": {f"{r['score'].get('kind')}#{i + 1}": x["pred"] for r in rows for i, x in enumerate(r["score"].get("rows", [])) if not x["ok"]},
        "bad_length": sum(1 for r in rows if r["score"].get("n_returned") != N),
    }
