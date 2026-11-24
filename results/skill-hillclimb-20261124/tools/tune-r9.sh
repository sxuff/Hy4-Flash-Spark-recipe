#!/usr/bin/env bash
# Recipe v2.1 re-measure with the recipe's own bench/tune.py (same arm specs as tune-r6.sh / tune-r7.sh, plus
# "extra": ["--spec-draft-p-min","0.75"]), v2 controls on both sides of the decode arm. tune.py's own `begin`/`end`
# stop and restore GLM, this box's incumbent when the recipe was written; today's incumbent is inkling2-mini.service,
# so this script drains/stops/restores that instead and writes begin.json the way `tune.py begin` does.
set -u
cd ~/.hermes/hy4-single-spark
export TUNE_RUN=tune-20261124
LAB=runs/$TUNE_RUN; mkdir -p "$LAB"
exec 9>/tmp/gb10-hillclimb.lock
flock -n 9 || { echo "another GPU queue holds /tmp/gb10-hillclimb.lock"; exit 1; }
restore() {
  systemctl --user reset-failed inkling2-mini.service 2>/dev/null
  systemctl --user start inkling2-mini.service
  systemctl --user stop inkling-fallback-restore.timer 2>/dev/null
  echo "$(date +%T) RESTORED inkling2-mini.service: $(systemctl --user is-active inkling2-mini.service)"
}
trap restore EXIT
systemd-run --user --on-active=4h --unit inkling-fallback-restore systemctl --user start inkling2-mini.service >/dev/null 2>&1
for i in $(seq 100); do
  r=$(curl -s -m 3 127.0.0.1:19020/metrics | awk '/^vllm:num_requests_running/{s+=$2} END{print s+0}')
  [ "${r:-0}" = 0 ] && break; sleep 3
done
echo "$(date +%T) stopping inkling2-mini.service (running requests: ${r:-0})"
systemctl --user stop inkling2-mini.service
for i in $(seq 60); do docker ps -q -f name='^inkling2-mini-gx10$' | grep -q . || break; sleep 2; done
sleep 5
python3 -c "
import json, time, qualify as Q
a, s = Q.mem()
json.dump({'time': time.time(), 'swap_baseline': s, 'avail_gib': a / 2**30,
           'note': 'written by run.sh in place of tune.py begin (incumbent: inkling2-mini.service)'}, open('$LAB/begin.json', 'w'))
print('begin.json: avail %.1f GiB, swap %d' % (a / 2**30, s))"
ENVJ='"GGML_CUDA_PDL":"1","GGML_CUDA_GRAPH_OPT":"1","GGML_CUDA_FA_576_WHOLE_TILES":"1"'
B='"ctx":262144,"depth":4,"ubatch":2048,"batch":4096,"load_mode":"dio"'
P='"extra":["--spec-draft-p-min","0.75"]'
arm() {
  echo "$(date +%T) === $1"
  systemd-run --user --scope -q -p MemorySwapMax=0 python3 tune.py arm "$1" "$2" > "$LAB/$1.log" 2>&1
  echo "$(date +%T) $1 exit $?"
  grep -E "loaded|temps|'panel'|panel |needle|ARM FAILURE|done " "$LAB/$1.log" | tail -9 | cut -c1-220
}
arm r9-d4-v2 "{$B,\"env\":{$ENVJ},\"work\":[\"temps\"],\"reps\":2}"
arm r9-d4-pmin75 "{$B,\"env\":{$ENVJ},$P,\"work\":[\"temps\"],\"reps\":2}"
arm r9-d4-v2b "{$B,\"env\":{$ENVJ},\"work\":[\"temps\"],\"reps\":2}"
arm r9-final-pmin75 "{$B,\"cache_ram\":0,\"env\":{$ENVJ},$P,\"work\":[\"panel\",\"prefill\",\"cache\",\"swap\",\"needle60000\"]}"
echo "$(date +%T) TUNE DONE"
