#!/bin/bash
# Ночь 1 домеров под методику: повтор S1-326 (все 10 моделей) и замер без кэша на 50 обращениях.
set -u
cd "$(dirname "$0")/.."
YA=hf.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF:Q4_K_M
TL=hf.co/t-tech/T-lite-it-2.1-GGUF:Q4_K_M
GC=hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:Q4_K_M
Q3=hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M
GL=hf.co/bartowski/THUDM_GLM-4-9B-0414-GGUF:Q4_K_M
QV=hf.co/Vikhrmodels/QVikhr-3-4B-Instruction-GGUF:Q4_K_M
ALL="$GC,$YA,$TL,$Q3,$GL,$QV,qwen3.5:4b,qwen3.5:2b,gemma3:4b,granite4:micro-h"
echo "== start $(date -u) $(nproc) vCPU $(docker exec ollama-ollama-1 ollama --version 2>&1 | tail -1)"
python3 runner.py scenarios/01b_class_big --tag ram8-r2 --models "$ALL"
echo "== repeat done $(date -u)"
python3 runner.py scenarios/01b_class_big --tag ram8-nocache50 --sample 50 --unique-prefix --models "$ALL"
echo "== end $(date -u)"
echo NIGHT3_DONE
