"""Сценарий 9. RAG: ответ по найденным фрагментам. Оценка — точность на вопросах с ответом и честное «не знаю»."""
import re

OPTIONS = {"num_ctx": 4096, "num_predict": 200}


def build_messages(item, ds):
    ctx = "\n\n".join(f"<<{c['id']}>>\n{c['text']}" for c in item["context"])
    system = (
        "Ты ассистент службы поддержки и сотрудников интернет-магазина «ТехноДом». Отвечай на вопрос ТОЛЬКО по приведённым "
        "фрагментам базы знаний. Если ответа во фрагментах нет — поставь found=false и напиши, что в базе знаний нет этой "
        "информации; ничего не придумывай. Ответ — 1–2 предложения по-русски. В source укажи id фрагмента, на который опираешься.\n"
        'Ответь только JSON: {"found": true/false, "answer": "...", "source": "..."}'
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Фрагменты базы знаний:\n\n{ctx}\n\nВопрос: {item['q']}"}]


def schema(ds):
    return {"type": "object", "properties": {"found": {"type": "boolean"}, "answer": {"type": "string"}, "source": {"type": "string"}},
            "required": ["found", "answer", "source"]}


def _n(s):
    return str(s or "").lower().replace("ё", "е")


NOINFO = r"нет (такой |этой |данной )?информаци|не указан|нет данных|не найд|не содерж|не упомина|не рассматрива|отсутству|не приводится|неизвестн"
REJECTED = r"не одобр|отклон|вместо|сначала|изначально|обсужда|предлага|отказ"


def score(item, parsed, raw, ds):
    """Оценка по тексту ответа. Флаг found учитываем отдельно: модели часто путают его с ответом «да/нет»."""
    p = parsed if isinstance(parsed, dict) else {}
    ans, found = _n(p.get("answer")), p.get("found")
    says_noinfo = bool(re.search(NOINFO, ans))
    if not item["answerable"]:
        # честно: прямо сказал, что информации нет; либо found=false и в ответе нет утверждений с цифрами
        ok = says_noinfo or (found is False and not re.search(r"\d", ans))
        verdict = "честно: нет данных" if ok else "выдумал ответ"
        flag_ok = found is False
    else:
        hit = all(re.search(x, ans) for x in item["all"]) if item.get("all") else any(re.search(x, ans) for x in item["any"])
        bad = any(re.search(x, ans) for x in item.get("forbid", [])) and not re.search(REJECTED, ans)
        bad = bad or any(ans.startswith(x) for x in item.get("forbid_start", []))
        ok = hit and not bad and not says_noinfo
        verdict = "верно" if ok else ("отказался, хотя ответ был" if says_noinfo else "неверно")
        flag_ok = found is True
    src_ok = None
    if item["answerable"] and item.get("source"):
        src_ok = item["source"] in str(p.get("source") or "")
    return {"ok": ok, "verdict": verdict, "type": item["type"], "answerable": item["answerable"], "found": found,
            "flag_consistent": flag_ok, "answer": p.get("answer"), "source": p.get("source"), "source_ok": src_ok,
            "retrieval_hit": item.get("retrieval_hit")}


def summarize(rows, ds):
    ans = [r for r in rows if r["score"]["answerable"]]
    non = [r for r in rows if not r["score"]["answerable"]]
    return {
        "answerable_correct": f"{sum(r['ok'] for r in ans)}/{len(ans)}",
        "honest_unknown": f"{sum(r['ok'] for r in non)}/{len(non)}",
        "false_refusals": sum(1 for r in ans if r["score"].get("verdict") == "отказался, хотя ответ был"),
        "flag_inconsistent": sum(1 for r in rows if r["score"].get("flag_consistent") is False),
        "source_correct": f"{sum(1 for r in ans if r['score'].get('source_ok'))}/{len(ans)}",
        "wrong": {r["id"]: r["score"].get("verdict", "ошибка запроса") for r in rows if not r["ok"]},
    }
