#!/usr/bin/env python3
"""Расширенный синтетический датасет для S1 (классификация обращений): ~300 обращений.
Генерация — deepseek-v4.1-flash по батчам (категория × сложность × стиль), затем разметка проверяется
двумя сильными моделями (deepseek-v4-pro, mimo-v2.6-pro) с тем же промптом, что и у тестируемых моделей.
Метка = большинство из трёх голосов (задуманная категория + две модели); если все три разные — пример выбрасывается.
  python3 gen_s1big.py gen      -> scenarios/01b_class_big/raw.json
  python3 gen_s1big.py verify   -> scenarios/01b_class_big/dataset.json"""
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import apirunner  # noqa: E402
import runner  # noqa: E402

SRC = HERE / "scenarios" / "01_classification"
DST = HERE / "scenarios" / "01b_class_big"
GEN_MODEL = "deepseek/deepseek-v4.1-flash"
CHECK_MODELS = ["deepseek/deepseek-v4-pro", "xiaomi/mimo-v2.6-pro"]

DIFF = {
    "easy": "однозначное обращение, цель клиента очевидна",
    "medium": "есть лишние детали, эмоции или упоминание другой темы, но главная цель понятна",
    "hard": "в обращении смешаны две темы (например, жалоба на курьера и вопрос о сроках, брак и возврат денег); "
            "категория определяется по ГЛАВНОЙ цели клиента — что он хочет получить в итоге",
}
STYLES = [
    "короткие сообщения из чата, без приветствия, иногда с опечатками и без знаков препинания",
    "развёрнутые письма на почту, вежливые, с номером заказа",
    "эмоциональные сообщения, раздражение, капслок местами",
    "сообщения от юрлиц и закупщиков, деловой стиль",
    "пожилые клиенты, многословно, путают термины",
    "голосовые сообщения, расшифрованные в текст: разговорный стиль, повторы, слова-паразиты",
]


def gen():
    ds = json.loads((SRC / "dataset.json").read_text())
    cats = ds["categories"]
    jobs = []
    k = 0
    for c in cats:
        for d, n_batches in (("easy", 2), ("medium", 2), ("hard", 2)):
            for b in range(n_batches):
                k += 1
                jobs.append((k, c, d, STYLES[(k + b) % len(STYLES)]))
    schema = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}}},
              "required": ["items"]}
    cat_list = "\n".join(f"- {kk}: {v}" for kk, v in cats.items())

    def one(job):
        k, c, d, style = job
        msg = [{"role": "user", "content": (
            f"Сгенерируй 9 разных реалистичных обращений клиентов в поддержку. Контекст: {ds['context']}\n"
            f"Категории поддержки:\n{cat_list}\n\n"
            f"Все 9 обращений должны относиться к категории «{c}» по главной цели клиента.\n"
            f"Сложность: {d} — {DIFF[d]}.\nСтиль: {style}.\n"
            "Разные товары, суммы, номера заказов, города, ситуации. Длина 1–5 предложений. Не упоминай названия "
            "категорий. Без персональных данных реальных людей. Ответ — JSON {\"items\": [\"текст\", ...]}.")}]
        p = {}
        for t in range(4):
            r = apirunner.chat(GEN_MODEL, msg, None, {"temperature": 0.9, "seed": 1000 + k + 100 * t, "num_predict": 2500}, False)
            p = runner.parse_json(r["content"]) or {}
            if len([x for x in p.get("items", []) if isinstance(x, str) and len(x) > 15]) >= 7:
                break
        return [{"label": c, "difficulty": d, "style": style, "text": t.strip()} for t in p.get("items", [])
                if isinstance(t, str) and len(t.strip()) > 15]

    prev = json.loads((DST / "raw.json").read_text()) if (DST / "raw.json").exists() else []
    have = Counter((i["label"], i["difficulty"]) for i in prev)
    need, jobs2 = Counter(), []
    for j in jobs:
        if have[(j[1], j[2])] + 9 * need[(j[1], j[2])] < 18:
            need[(j[1], j[2])] += 1
            jobs2.append(j)
    print("batches to run:", len(jobs2))
    with ThreadPoolExecutor(6) as ex:
        out = prev + [x for batch in ex.map(one, jobs2) for x in batch]
    seen, uniq = set(), []
    for it in out:
        key = it["text"].lower()[:60]
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    DST.mkdir(exist_ok=True)
    (DST / "raw.json").write_text(json.dumps(uniq, ensure_ascii=False, indent=1))
    print("generated", len(out), "unique", len(uniq), Counter(i["label"] for i in uniq), f"spent {apirunner.spent():.2f}")


def verify():
    spec = runner.load_scenario(SRC)
    mod, ds = spec
    raw = json.loads((DST / "raw.json").read_text())
    schema = mod.schema(ds)

    def vote(args):
        i, it, m = args
        r = apirunner.chat(m, mod.build_messages(it, ds), schema, {"temperature": 0, "seed": 42, "num_predict": 32}, False)
        p = runner.parse_json(r["content"]) or {}
        return i, m, p.get("category")

    tasks = [(i, it, m) for i, it in enumerate(raw) for m in CHECK_MODELS]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(vote, tasks))
    votes = {}
    for i, m, c in res:
        votes.setdefault(i, {})[m] = c
    items, dropped, relabeled, agree_all = [], 0, 0, 0
    for i, it in enumerate(raw):
        v = [it["label"]] + [votes[i].get(m) for m in CHECK_MODELS]
        top, cnt = Counter(v).most_common(1)[0]
        if cnt < 2 or top is None:
            dropped += 1
            continue
        agree_all += cnt == 3
        relabeled += top != it["label"]
        items.append({"id": len(items) + 1, "label": top, "intended": it["label"], "difficulty": it["difficulty"],
                      "votes": v, "text": it["text"]})
    out = {k: v for k, v in ds.items() if k != "items"}
    out.update({"scenario": "01b_class_big", "title": "Классификация обращений — расширенная выборка",
                "provenance": f"синтетика {GEN_MODEL}, разметка — большинство из (задумано, {', '.join(CHECK_MODELS)})",
                "items": items})
    (DST / "dataset.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"kept {len(items)} dropped {dropped} relabeled {relabeled} unanimous {agree_all}",
          Counter(i["label"] for i in items), Counter(i["difficulty"] for i in items), f"spent {apirunner.spent():.2f}")


if __name__ == "__main__":
    {"gen": gen, "verify": verify}[sys.argv[1]]()
