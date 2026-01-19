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

# This script clones a fresh copy of the repo, installs it, runs benchmarks on any new commits its finds. The script starts with
# the oldest commit in the log that wasn't previously run, and continues until it has run all commits.  By default, it runs all
# benchmarks defined in config.txt.

# New benchmark results are appended to a json file in a configurable branch (typically `gh-pages`) and pushed to github.

# This script is intended to be run as a nightly cron job. It expects an ssh identify file to be present at
# ~/.ssh/id_ed25519_mujoco_warp_nightly that correspodns to a deploy key on github with write access to
# https://github.com/google-deepmind/mujoco_warp

set -euo pipefail

REPO_URL="git@github.com:google-deepmind/mujoco_warp.git"
RESULTS_BRANCH="gh-pages"
WORK_DIR="/tmp/mujoco_warp-nightly-$$"
RESULTS_DIR="/tmp/mujoco_warp-results-$$"
RESULTS_DIR_PATH="$RESULTS_DIR/nightly"
LAST_COMMIT_FILE="$RESULTS_DIR_PATH/last_commit.txt"

# Put UV cache in /tmp so hardlinking works (same filesystem as work directory)
export UV_CACHE_DIR="/tmp/uv-cache-nightly-$$"

log() {
    ( [ -n "${1:-}" ] && echo "$@" || cat ) | while read -r l; do
        printf "[%(%Y-%m-%d %H:%M:%S)T] %s\n" -1 "$l"
    done
}

error() {
    log "ERROR: $*" >&2
    exit 1
}

clear_gpu() {
    local gpu_device="$1"
    log "Clearing GPU $gpu_device..."
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$gpu_device" 2>/dev/null || true)
    if [[ -n "$gpu_pids" ]]; then
        for pid in $gpu_pids; do
            log "Killing process $pid on GPU $gpu_device"
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 2  # Wait for processes to fully terminate
    fi
}

cleanup() {
    if [[ -d "$WORK_DIR" ]]; then
        log "Cleaning up work directory..."
        rm -rf "$WORK_DIR"
    fi
    if [[ -d "$RESULTS_DIR" ]]; then
        log "Cleaning up results directory..."
        rm -rf "$RESULTS_DIR"
    fi
}
# trap cleanup EXIT

export GIT_SSH_COMMAND="ssh -i \"$HOME/.ssh/id_ed25519_mujoco_warp_nightly\" \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=accept-new"

# Clone fresh copy of mujoco_warp
log "Cloning mujoco_warp..."
git clone "$REPO_URL" "$WORK_DIR"
cd "$WORK_DIR"

# Clone results branch (shallow clone for speed)
log "Cloning results branch..."
git clone --branch "$RESULTS_BRANCH" --depth 1 "$REPO_URL" "$RESULTS_DIR"

# Extract the most recent commit SHA from last_commit.txt
if [[ ! -f "$LAST_COMMIT_FILE" ]]; then
    error "No existing commit tracker found at $LAST_COMMIT_FILE"
fi

LAST_COMMIT=$(cat "$LAST_COMMIT_FILE")
if [[ -z "$LAST_COMMIT" ]]; then
    error "Failed to read last commit from $LAST_COMMIT_FILE"
fi
log "Last benchmarked commit: $LAST_COMMIT"

# Get list of commits from LAST_COMMIT to HEAD
cd "$WORK_DIR"
COMMITS=$(git rev-list --reverse "${LAST_COMMIT}..HEAD")

# Loop through each commit
for commit in $COMMITS; do
    log "Processing commit $commit"
    
    # Checkout the commit
    git checkout "$commit" 2>&1 | log
    
    # Get commit timestamp
    COMMIT_TIMESTAMP=$(TZ=UTC git log -1 --format=%cd --date=format-local:'%Y-%m-%dT%H:%M:%S+00:00' "$commit")
    
    # Load config.txt and run benchmarks
    CONFIG="$WORK_DIR/benchmarks/config.txt"
    if [[ ! -f "$CONFIG" ]]; then
        error "Configuration file not found at $CONFIG"
    fi
    
    while read -r NAME MJCF NWORLD NCONMAX NJMAX NSTEP REPLAY; do
        # Skip comments and empty lines
        [[ "$NAME" =~ ^#.*$ || -z "$NAME" ]] && continue
        
        log "Running benchmark: $NAME"
        
        # Build command arguments for mjwarp-testspeed
        CMD=(
            "mjwarp-testspeed"
            "$WORK_DIR/benchmarks/$MJCF"
            "--nworld=$NWORLD"
            "--nconmax=$NCONMAX"
            "--njmax=$NJMAX"
            "--clear_kernel_cache=false"
            "--format=json"
            "--event_trace=true"
            "--memory=true"
            "--measure_solver=true"
            "--measure_alloc=true"
        )
        [[ "$NSTEP" != "-" ]] && CMD+=( "--nstep=$NSTEP" )
        [[ "$REPLAY" != "-" ]] && CMD+=( "--replay=$REPLAY" )
        
        # Run benchmark using uv run (handles venv and dependencies automatically)
        # --prerelease=allow: accept dev/nightly builds
        # --upgrade: always get the latest versions
        BENCHMARK_JSON=$(UV_NO_CONFIG=1 uv run --prerelease=allow --upgrade "${CMD[@]}")
        
        # Convert multi-line JSON to single line and add commit metadata
        RESULT=$(echo "$BENCHMARK_JSON" | python3 -c "
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
    
    # Update last_commit.txt to track progress
    echo "$commit" > "$LAST_COMMIT_FILE"
    log "Updated last commit tracker to $commit"
    
    log "Finished processing commit $commit"
done

# Commit and push results to gh-pages branch
log "Committing and pushing benchmark results..."
cd "$RESULTS_DIR"

# Stage all changes (JSONL files and last_commit.txt)
git add nightly/*.jsonl nightly/last_commit.txt

# Check if there are changes to commit
if git diff --staged --quiet; then
    log "No changes to commit"
else
    # Create commit message with date and commit range
    COMMIT_MSG="Update nightly benchmarks - $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    git commit -m "$COMMIT_MSG"
    
    # Push to gh-pages
    git push origin "$RESULTS_BRANCH"
    log "Successfully pushed results to $RESULTS_BRANCH"
fi

log "Nightly benchmarking complete!"
