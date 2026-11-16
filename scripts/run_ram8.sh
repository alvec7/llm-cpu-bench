#!/bin/bash
# Шаг 1 после апгрейда до 5 vCPU / 8 ГБ: те же модели, конфигурация Ollama A (как в шаблоне).
set -u
cd "$(dirname "$0")/.."
Q=hf.co/Vikhrmodels/QVikhr-3-4B-Instruction-GGUF:Q4_K_M
M4="$Q,qwen3.5:4b,qwen3.5:2b,gemma3:4b"
echo "== host: $(nproc) vCPU, $(free -g | awk '/Mem/{print $2}') GB"
python3 runner.py scenarios/01_classification --tag ram8 --models "$M4"
python3 runner.py scenarios/04_summary --tag ram8 --models "$M4"
python3 runner.py scenarios/09_rag --tag ram8 --models "$M4,granite4:micro-h"
for m in "$Q" qwen3.5:4b qwen3.5:2b; do
  python3 loadtest.py --model "$m" --profile short --conc 1,2,4 --n 12 --tag A8
  python3 loadtest.py --model "$m" --profile medium --conc 1,2,4 --n 8 --tag A8
done
echo RAM8_DONE
