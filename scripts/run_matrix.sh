#!/usr/bin/env bash
# Run the quality matrix: for each precision, start the server, run every
# condition on the chosen split, stop the server. Resumable: 03 skips questions
# already answered, so rerunning after a crash continues where it stopped.
#
#   bash scripts/run_matrix.sh                 # bf16 then int4, 4 conditions, final split -> results/final
#   bash scripts/run_matrix.sh int4            # one precision
#   DRY_RUN=1 bash scripts/run_matrix.sh       # print the commands only
#   CONDITIONS="clean Fog" SPLIT=development OUT_DIR=results/dev N=20 bash scripts/run_matrix.sh
#   INT4_MODEL=checkpoints/qwen2.5-vl-7b-awq-w4a16-g128 bash scripts/run_matrix.sh int4   # own checkpoint
#   RSYNC_TO=user@host:/path/qflip bash scripts/run_matrix.sh   # copy results/ and manifests/ after each run
#
# Logs: logs/matrix_<timestamp>.txt (this driver), logs/serve_<precision>_<timestamp>.txt (vLLM),
# and 03's own output appended to logs/matrix_<precision>_<condition>.txt.
set -euo pipefail
cd "$(dirname "$0")/.."

PRECISIONS=("$@")
[[ ${#PRECISIONS[@]} -eq 0 ]] && PRECISIONS=(bf16 int4)
CONDITIONS="${CONDITIONS:-clean NoImage LowLight Fog}"
SPLIT="${SPLIT:-final}"
OUT_DIR="${OUT_DIR:-results/final}"
N="${N:-0}"
PORT="${PORT:-8000}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1200}"
DRY_RUN="${DRY_RUN:-0}"
RSYNC_TO="${RSYNC_TO:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/matrix_${STAMP}.txt"
export QFLIP_BASE_URL="http://localhost:${PORT}/v1"
export PORT

if [[ "$DRY_RUN" == "1" ]]; then LOG=/dev/null; else mkdir -p logs "$OUT_DIR"; fi
log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
run() { if [[ "$DRY_RUN" == "1" ]]; then echo "+ $*"; else log "+ $*"; "$@"; fi; }

wait_healthy() {
  local pid=$1 waited=0
  while ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then log "server exited before becoming healthy"; return 1; fi
    if (( waited >= HEALTH_TIMEOUT_S )); then log "server not healthy after ${HEALTH_TIMEOUT_S}s"; return 1; fi
    sleep 5; waited=$((waited + 5))
  done
  log "server healthy after ${waited}s"
}

stop_server() {
  local pid=$1
  if kill -0 "$pid" 2>/dev/null; then
    log "stopping server pid $pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
  sleep 5
}

sync_out() {
  [[ -z "$RSYNC_TO" ]] && return 0
  run rsync -a --exclude '*.tmp' results/ manifests/ logs/ "$RSYNC_TO/" 2>/dev/null \
    || run rsync -a results "$RSYNC_TO/" && run rsync -a manifests logs "$RSYNC_TO/"
}

log "matrix start: precisions=${PRECISIONS[*]} conditions=${CONDITIONS} split=${SPLIT} out=${OUT_DIR} n=${N} port=${PORT}"
[[ "$DRY_RUN" == "1" ]] && echo "+ (dry run; nothing executed)"

# Final split: the core freeze must exist and match the live config before any server starts.
if [[ "$SPLIT" == "final" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ python scripts/freeze_config.py --check   (required for SPLIT=final)"
  else
    python scripts/freeze_config.py --check || { log "final split refused: freeze check failed"; exit 1; }
  fi
fi

port_free() { ! (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":${PORT} "; }
served_name() { python -c "import yaml; print(yaml.safe_load(open('config/pilot.yaml'))['models']['$1'])"; }
server_serves() { curl -sf "http://localhost:${PORT}/v1/models" | python -c "import sys,json; ids=[m['id'] for m in json.load(sys.stdin)['data']]; sys.exit(0 if '$1' in ids else 1)"; }

for p in "${PRECISIONS[@]}"; do
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ port ${PORT} must be free"
    echo "+ bash scripts/01_serve.sh $p  > logs/serve_${p}_${STAMP}.txt 2>&1 &"
    echo "+ wait for http://localhost:${PORT}/health, then /v1/models must list $(served_name "$p"); manifest pid and port listener must be that server"
    [[ "$SPLIT" == "final" ]] && echo "+ python scripts/freeze_config.py --check --serve-manifest manifests/serve_${p}.txt --precision $p"
  else
    port_free || { log "port ${PORT} already in use: another server is running; refusing to reuse it"; exit 1; }
    log "starting server: $p"
    bash scripts/01_serve.sh "$p" > "logs/serve_${p}_${STAMP}.txt" 2>&1 &
    SERVER_PID=$!
    trap 'stop_server $SERVER_PID' EXIT
    wait_healthy "$SERVER_PID" || { tail -20 "logs/serve_${p}_${STAMP}.txt"; exit 1; }
    want="$(served_name "$p")"
    server_serves "$want" || { log "server on port ${PORT} does not serve ${want}"; exit 1; }
    man_pid="$(sed -n 's/^pid: //p' "manifests/serve_${p}.txt")"
    [[ "$man_pid" == "$SERVER_PID" ]] || { log "manifests/serve_${p}.txt pid ${man_pid:-none} is not the server this driver started (${SERVER_PID}); another 01_serve.sh overwrote it"; exit 1; }
    listener_pid="$(ss -ltnp 2>/dev/null | grep ":${PORT} " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)"
    [[ -n "$listener_pid" ]] || { log "cannot determine which process listens on port ${PORT} (ss -ltnp); refusing to run without listener ownership"; exit 1; }
    listener_ppid="$(awk '/^PPid/{print $2}' "/proc/${listener_pid}/status" 2>/dev/null || true)"
    if [[ "$listener_pid" != "$SERVER_PID" && "$listener_ppid" != "$SERVER_PID" ]]; then
      log "port ${PORT} is held by pid ${listener_pid} (parent ${listener_ppid:-?}), not the started server ${SERVER_PID}"; exit 1
    fi
    if [[ "$SPLIT" == "final" ]]; then
      python scripts/freeze_config.py --check --serve-manifest "manifests/serve_${p}.txt" --precision "$p" \
        || { log "final split refused: server manifest differs from the freeze"; exit 1; }
    fi
  fi

  for c in $CONDITIONS; do
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "+ python scripts/03_smoke_drivebench.py --precision $p --condition $c --n $N --split $SPLIT --out-dir $OUT_DIR  |& tee -a logs/matrix_${p}_${c}.txt"
    else
      log "run: $p $c"
      python scripts/03_smoke_drivebench.py --precision "$p" --condition "$c" --n "$N" --split "$SPLIT" --out-dir "$OUT_DIR" 2>&1 | tee -a "logs/matrix_${p}_${c}.txt"
      cp "manifests/serve_${p}.txt" "$OUT_DIR/serve_${p}.txt"
      sync_out
    fi
  done

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ kill server ($p)"
  else
    stop_server "$SERVER_PID"
    trap - EXIT
  fi
done

log "matrix done"
[[ "$DRY_RUN" == "1" ]] || echo "next: python scripts/05_compare_flips.py $(for c in $CONDITIONS; do printf '%s/bf16_%s.jsonl %s/int4_%s.jsonl ' "$OUT_DIR" "$c" "$OUT_DIR" "$c"; done)--by cell n_views margin condition"
