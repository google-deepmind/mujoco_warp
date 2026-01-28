#!/bin/bash

# Copyright 2026 The Newton Developers
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

# Backfill script for generating historical benchmark data.
#
# Unlike nightly.sh, this script:
# - Uses testspeed.py and config.txt from HEAD (not each commit's version)
# - Uses a local python environment instead of uv run
# - Does NOT update last_commit.txt or push to git
# - Targets a range of commits by number (HEAD~start to HEAD~end)
#
# Usage:
#   ./backfill.sh [--mock] [--start=N] [--end=N]
#
# Options:
#   --mock      Run benchmarks with nworld=1 and nstep=10 for fast testing
#   --start=N   Start from HEAD~N (default: 100, the oldest commit)
#   --end=N     End at HEAD~N (default: 1, most recent)
#
# Examples:
#   ./backfill.sh                    # Process HEAD~100 to HEAD~1
#   ./backfill.sh --start=50         # Process HEAD~50 to HEAD~1
#   ./backfill.sh --start=100 --end=90  # Process only HEAD~100 to HEAD~90

set -euo pipefail

# Script location (HEAD checkout with current testspeed.py and config.txt)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEAD_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration - clone from local dir to avoid SSH key issues and for speed
WORK_DIR="/tmp/mujoco_warp-backfill-$$"
RESULTS_DIR_PATH="$HEAD_DIR/../mujoco_warp_gh_pages/nightly"
CONFIG="$SCRIPT_DIR/config.txt"

# Default options
START=100
END=1
MOCK_MODE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --mock)
            MOCK_MODE=true
            shift
            ;;
        --start=*)
            START="${arg#*=}"
            shift
            ;;
        --end=*)
            END="${arg#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--mock] [--start=N] [--end=N]"
            exit 1
            ;;
    esac
done

log() {
    ( [ -n "${1:-}" ] && echo "$@" || cat ) | while read -r l; do
        printf "[%(%Y-%m-%d %H:%M:%S)T] %s\n" -1 "$l"
    done
}

error() {
    log "ERROR: $*" >&2
    exit 1
}

cleanup() {
    if [[ -d "$WORK_DIR" ]]; then
        log "Cleaning up work directory..."
        rm -rf "$WORK_DIR"
    fi
}
trap cleanup EXIT

# Verify environment
if [[ ! -f "$CONFIG" ]]; then
    error "Config file not found at $CONFIG"
fi
if [[ ! -d "$RESULTS_DIR_PATH" ]]; then
    error "Results directory not found at $RESULTS_DIR_PATH"
fi

log "Backfill configuration:"
log "  RANGE: HEAD~$START to HEAD~$END"
log "  MOCK_MODE: $MOCK_MODE"
log "  HEAD_DIR: $HEAD_DIR"
log "  RESULTS_DIR: $RESULTS_DIR_PATH"

# Clone from local directory (faster than GitHub, no SSH key needed)
log "Cloning mujoco_warp to work directory..."
git clone "$HEAD_DIR" "$WORK_DIR"

# Get list of commits in range (oldest first)
cd "$HEAD_DIR"
COMMITS=$(git rev-list --reverse "HEAD~${START}..HEAD~${END}")
TOTAL_COMMITS=$(echo "$COMMITS" | wc -l)
log "Found $TOTAL_COMMITS commits to process"

CURRENT=0
for commit in $COMMITS; do
    CURRENT=$((CURRENT + 1))
    log "[$CURRENT/$TOTAL_COMMITS] Processing commit $commit"
    
    # Checkout the commit in work directory (force to handle uv.lock changes)
    cd "$WORK_DIR"
    git checkout --force "$commit" 2>&1 | log
    
    # Copy HEAD's testspeed.py into the work directory (overwrite historical version)
    cp "$HEAD_DIR/mujoco_warp/testspeed.py" "$WORK_DIR/mujoco_warp/testspeed.py"
    
    # Get commit timestamp
    COMMIT_TIMESTAMP=$(TZ=UTC git log -1 --format=%cd --date=format-local:'%Y-%m-%dT%H:%M:%S+00:00' "$commit")
    
    # Run benchmarks using HEAD's config.txt
    while read -r NAME MJCF NWORLD NCONMAX NJMAX NSTEP REPLAY; do
        # Skip comments and empty lines
        [[ "$NAME" =~ ^#.*$ || -z "$NAME" ]] && continue
        
        log "Running benchmark: $NAME"
        
        # In mock mode, use small nworld and nstep for speed
        if [[ "$MOCK_MODE" == "true" ]]; then
            NWORLD=1
            NSTEP=10
        fi
        
        # Build command arguments for mjwarp-testspeed
        CMD=(
            "mjwarp-testspeed"
            "$SCRIPT_DIR/$MJCF"
            "--nworld=$NWORLD"
            "--nconmax=$NCONMAX"
            "--njmax=$NJMAX"
            "--clear_warp_cache=true"
            "--format=json"
            "--event_trace=true"
            "--memory=true"
            "--measure_solver=true"
            "--measure_alloc=true"
        )
        [[ "$NSTEP" != "-" ]] && CMD+=( "--nstep=$NSTEP" )
        [[ "$REPLAY" != "-" ]] && CMD+=( "--replay=$REPLAY" )
        
        # Run benchmark using uv run from WORK_DIR (historical commit + HEAD's testspeed.py)
        # Send uv's stderr to /dev/null (or log), capture only stdout (JSON)
        cd "$WORK_DIR"
        log "Command: UV_NO_CONFIG=1 uv run --prerelease=allow --upgrade ${CMD[*]}"
        if ! BENCHMARK_JSON=$(UV_NO_CONFIG=1 uv run --prerelease=allow --upgrade "${CMD[@]}" 2>/dev/null); then
            log "WARNING: Benchmark $NAME failed for commit $commit, retrying with verbose output..."
            UV_NO_CONFIG=1 uv run --prerelease=allow --upgrade "${CMD[@]}" 2>&1 | log
            continue
        fi

        echo "$BENCHMARK_JSON"
        
        # Convert multi-line JSON to single line and add commit metadata
        # Use tail -n 1 to ignore any log spam before the JSON output
        RESULT=$(echo "$BENCHMARK_JSON" | tail -n 1 | python3 -c "
import sys, json
data = json.load(sys.stdin)
data['commit'] = '$commit'
data['timestamp'] = '$COMMIT_TIMESTAMP'
print(json.dumps(data))
")
        
        # Append to benchmark-specific JSONL file
        BENCHMARK_FILE="$RESULTS_DIR_PATH/${NAME}.jsonl"
        printf "%s\n" "$RESULT" >> "$BENCHMARK_FILE"
        log "Benchmark $NAME completed"
    done < "$CONFIG"
    
    log "Finished processing commit $commit"
done

log "Backfill complete!"
log "Results written to: $RESULTS_DIR_PATH"
log ""
log "Next steps:"
log "  1. Review the JSONL files for correctness"
log "  2. If satisfied, commit and push the results manually:"
log "     cd $RESULTS_DIR_PATH/.."
log "     git add nightly/*.jsonl"
log "     git commit -m 'Backfill benchmarks for HEAD~$START to HEAD~$END'"
log "     git push origin gh-pages"
