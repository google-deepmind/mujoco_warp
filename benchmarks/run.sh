#!/bin/bash

# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Run some or all benchmarks defined in this directory, and optionally store the results in a json file.

set -euo pipefail

log() {
  ( [ -n "${1:-}" ] && echo "$@" || cat ) | while read -r l; do
    printf "[%(%Y-%m-%d %H:%M:%S)T] %s\n" -1 "$l"
  done
}

error() {
  log "ERROR: $*" >&2
  exit 1
}

usage() {
  echo "run.sh: Runs MuJoCo Warp benchmarks."
  echo "Usage: $0 [-f benchmark-regex]"
  exit 1
}

FILTER=""
CLEAR_KERNEL_CACHE="false"

while [[ $# -gt 0 ]]; do
  # if the argument contains '=', split it into key and value
  if [[ "$1" == --*=* ]]; then
    set -- "${1%%=*}" "${1#*=}" "${@:2}"
  fi
  case $1 in
    --clear_kernel_cache) CLEAR_KERNEL_CACHE="$2"; shift 2 ;;
    -f|--filter) FILTER="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) error "Unknown option: $1" ;;
  esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG="$SCRIPT_DIR/config.txt"

if [[ ! -f "$CONFIG" ]]; then
    error "configuration file not found at $CONFIG"
fi

if ! command -v mjwarp-testspeed &> /dev/null; then
  error "mjwarp-testspeed not found. Please install MuJoCo Warp (or activate its environment)."
fi

while read -r NAME MJCF NWORLD NCONMAX NJMAX NSTEP REPLAY; do
    # Skip comments and empty lines
    [[ "$NAME" =~ ^#.*$ || -z "$NAME" ]] && continue
    
    # Apply filter
    [[ -n "$FILTER" && ! "$NAME" =~ $FILTER ]] && continue
    
    CMD=(
      "mjwarp-testspeed"
      "$SCRIPT_DIR/$MJCF"
      "--nworld=$NWORLD"
      "--nconmax=$NCONMAX"
      "--njmax=$NJMAX"
      "--clear_kernel_cache=$CLEAR_KERNEL_CACHE"
      "--format=short"
      "--event_trace=true"
      "--memory=true"
      "--measure_solver=true"
      "--measure_alloc=true"
    )
    [[ "$NSTEP" != "-" ]] && CMD+=( "--nstep=$NSTEP" )
    [[ "$REPLAY" != "-" ]] && CMD+=( "--replay=$REPLAY" )

    log "${CMD[@]}"

    "${CMD[@]}" | log
done < "$CONFIG"