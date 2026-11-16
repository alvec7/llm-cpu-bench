#!/bin/bash
# Вторая часть: параллельность на модели, которая её поддерживает (QVikhr, arch qwen3).
source <(sed -n '/^set -u/,/^run()/p' run_load.sh)
Q=hf.co/Vikhrmodels/QVikhr-3-4B-Instruction-GGUF:Q4_K_M
for CFG in A B C; do set_cfg $CFG; run $Q $CFG; done
set_cfg C; python3 loadtest.py --model qwen3.5:4b --profile short --conc 1,4 --n 12 --tag C
set_cfg A
echo LOAD_DONE
