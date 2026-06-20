#!/usr/bin/env bash
#
# TEMPORARY benchmark script for the nv-compact branch (delete before merge).
#
# Collects time-per-step / steps-per-second for the three solver configs on aloha_clutter,
# over two trajectory lengths:
#   configs:  none    - full solve, no sleeping, no islands (today's default baseline)
#             island  - island solver + sleeping (the old approach; via --enable_islands + SLEEP)
#             compact - active-DOF compaction + sleeping (this branch; via --nvmax + SLEEP)
#   lengths:  1k      - first 1000 replay steps (grippers in free space, almost all asleep)
#             full    - the whole pick replay (2751 steps, includes manipulation, more awake)
#
# Run from the repo root:   bash benchmarks/bench_compact.sh
# Sweep nworld:             NWORLDS="2048 8192" bash benchmarks/bench_compact.sh
#
# The hardware-independent number is the RATIO (none_step / compact_step); absolute steps/sec
# is GPU-bound. Lower "step" (ns/world-step) is better.

set -uo pipefail

NWORLDS="${NWORLDS:-8192}"     # space-separated list of nworld values
NVMAX="${NVMAX:-64}"           # compact block size; active set peaks ~52, so 64 covers it
SLEEP='opt.enableflags=SLEEP'  # enables sleeping -> compact (Newton) or island (with --enable_islands)

ASSET=/tmp/benchmark_assets/aloha_clutter
SCENE="$ASSET/scene_clutter.xml"
REPLAY="$ASSET/pick_clutter.npz"
BUF="--nconmax=512 --nccdmax=64 --njmax=1024"   # benchmark default buffers
OUT=/tmp/bench_compact_out
mkdir -p "$OUT"

# assemble assets once if missing (also runs the default benchmark; ignore its output)
if [[ ! -f "$SCENE" ]]; then
  echo "Assets not found, assembling via benchmarks/run.py (one-time, slow)..."
  uv run python benchmarks/run.py -f aloha_clutter --clear_warp_cache=false >/dev/null 2>&1 || true
fi
[[ -f "$SCENE" ]] || { echo "ERROR: $SCENE not found (asset setup failed)"; exit 1; }

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"

run() {  # $1=config  $2=len(1k|full)  $3=nstep_flag   rest: extra testspeed flags
  local cfg="$1" len="$2" nstep="$3"; shift 3
  local f="$OUT/${cfg}_${len}_nw${NW}.txt"
  if ! uv run mjwarp-testspeed "$SCENE" --nworld="$NW" $BUF $nstep \
        --replay="$REPLAY" --event_trace=true "$@" > "$f" 2>&1; then
    printf "  %-8s %-4s  FAILED (see %s)\n" "$cfg" "$len" "$f"; return
  fi
  local step sps solve coll ovf
  step=$(grep "Total time per step" "$f" | grep -oE "[0-9.]+")
  sps=$(grep "Total steps per second" "$f" | grep -oE "[0-9,]+" | tr -d ,)
  solve=$(awk '/^ *solve:/{print $2; exit}' "$f")
  coll=$(awk '/^ *collision:/{gsub(/[][,]/,""); s=0; for(i=2;i<=NF;i++)s+=$i; printf "%.1f", s; exit}' "$f")
  ovf=$(grep -c "overflow" "$f")
  printf "  %-8s %-4s  step=%-10s sps=%-9s solve=%-9s collision=%-9s overflow=%s\n" \
    "$cfg" "$len" "${step:-NA}" "${sps:-NA}" "${solve:-NA}" "${coll:-NA}" "$ovf"
}

for NW in $NWORLDS; do
  echo ""
  echo "================= nworld=$NW (step = ns/world-step, lower better) ================="
  run none    full ""
  run island  full ""             --enable_islands -o "$SLEEP"
  run compact full ""             --nvmax="$NVMAX" -o "$SLEEP"
  run none    1k   "--nstep=1000"
  run island  1k   "--nstep=1000" --enable_islands -o "$SLEEP"
  run compact 1k   "--nstep=1000" --nvmax="$NVMAX" -o "$SLEEP"
done

echo ""
echo "Raw testspeed output (with full event traces) saved under $OUT/"
echo "Speedup ratio = none.step / compact.step  (hardware-independent)."
