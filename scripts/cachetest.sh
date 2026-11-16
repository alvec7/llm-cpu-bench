#!/bin/bash
# Проверка ограничения кэша промптов llama-server через LLAMA_ARG_CACHE_RAM. Стартует после night4.
set -u
cd "$(dirname "$0")/.."
OV=/opt/ollama/docker-compose.bench.yml
compose() { (cd /opt/ollama && docker compose -f docker-compose.yml -f docker-compose.no-pull.yml ${1:+-f $OV} up -d ollama 2>&1 | tail -2); }
wait_ready() {
  for i in $(seq 1 60); do
    IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}' ollama-ollama-1 | awk '{print $1}')
    curl -s -m 2 "http://$IP:11434/api/version" >/dev/null && return 0; sleep 2
  done; echo "Ollama не поднялась"; return 1
}
echo "== start $(date -u)"
docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' ollama-ollama-1 | sort > logs/env_before.txt
[ -f $OV ] && { echo "override уже есть:"; cat $OV; }
printf 'services:\n  ollama:\n    environment:\n      LLAMA_ARG_CACHE_RAM: "1024"\n' > $OV
compose 1; wait_ready
echo "== env: $(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' ollama-ollama-1 | grep LLAMA_ARG)"
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 loadtest2.py --model qwen3.5:4b --profile short --conc 1,2,4 --cache nocache --min-s 300 --min-n 40 --rep 1 --tag cacheram1g
echo "== cache limits in log:"
docker logs --since "$T0" ollama-ollama-1 2>&1 | grep "cache state" | tail -3
echo "== oom: $(dmesg | grep -c 'Killed process')"
rm -f $OV; compose; wait_ready
docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' ollama-ollama-1 | sort > logs/env_after.txt
diff logs/env_before.txt logs/env_after.txt && echo "== env restored"
echo "== end $(date -u)"
echo CACHETEST_DONE
