#!/bin/bash
# Драйвер нагрузочного теста: конфигурации Ollama A/B/C × модели × профили.
# A — как было (без override), B — OLLAMA_NUM_PARALLEL=4, C — B + flash attention + KV-кэш q8_0.
set -u
cd "$(dirname "$0")/.."
OV=/opt/ollama/docker-compose.bench.yml
compose() { (cd /opt/ollama && docker compose -f docker-compose.yml -f docker-compose.no-pull.yml ${1:+-f $OV} up -d ollama >/dev/null 2>&1); }
wait_ready() {
  for i in $(seq 1 60); do
    IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}' ollama-ollama-1 | awk '{print $1}')
    curl -s -m 2 "http://$IP:11434/api/version" >/dev/null && return 0; sleep 2
  done; echo "Ollama не поднялась"; return 1
}
set_cfg() {  # $1 = A|B|C
  case $1 in
    A) rm -f $OV; compose "";;
    B) printf 'services:\n  ollama:\n    environment:\n      OLLAMA_NUM_PARALLEL: "4"\n' > $OV; compose 1;;
    C) printf 'services:\n  ollama:\n    environment:\n      OLLAMA_NUM_PARALLEL: "4"\n      OLLAMA_FLASH_ATTENTION: "1"\n      OLLAMA_KV_CACHE_TYPE: "q8_0"\n' > $OV; compose 1;;
  esac
  wait_ready && echo "== config $1: $(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' ollama-ollama-1 | grep -E 'PARALLEL|FLASH|KV_CACHE' | tr '\n' ' ')"
}
run() { python3 loadtest.py --model "$1" --profile short --conc 1,2,4 --n 12 --tag "$2"; python3 loadtest.py --model "$1" --profile medium --conc 1,2,4 --n 8 --tag "$2"; }
for CFG in A B C; do
  set_cfg $CFG
  run qwen3.5:4b $CFG
  [ "$CFG" != B ] && run qwen3.5:2b $CFG
done
set_cfg A   # вернуть исходные настройки
echo LOAD_DONE
