# MuJoCo Warp Benchmark Suite

MJWarp includes a collection of benchmarks for measuring performance across different robot models and scenarios.

## Installation

Make sure you have MuJoCo Warp installed for development:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

## Running Benchmarks

To execute all benchmarks, from the `benchmarks` directory run:

```bash
./run.sh
```

This will run all benchmarks defined in `config.txt` and output metrics in a columnar format.

### Filtering Benchmarks

To run specific benchmarks, use the `-f` or `--filter` option with a regex pattern:

```bash
# Run only the humanoid benchmark
./run.sh -f humanoid

# Run all Apollo variants
./run.sh -f apollo

# Run all benchmarks with "cloth" in the name
./run.sh -f cloth
```

### Clearing the Kernel Cache

By default, benchmarks use Warp's kernel cache for faster execution. To measure accurate JIT compilation times, you can disable the cache:

```bash
./run.sh --clear_kernel_cache=true
```

## Output Format

The benchmark script uses `mjwarp-testspeed` with the `--format=short` option, which outputs metrics in a columnar format:

```
2026-01-12 18:57:20] mjwarp-testspeed /home/<username>/mujoco_warp/benchmarks/humanoid/humanoid.xml --nworld=8192 --nconmax=24 --njmax=64 --clear_kernel_cache=false --format=short --event_trace=true --memory=true --measure_solver=true --measure_alloc=true
[2026-01-12 18:57:26] humanoid:jit_duration                                              0.3430611090734601
[2026-01-12 18:57:26] humanoid:run_time                                                  3.0016206190921366
[2026-01-12 18:57:26] humanoid:steps_per_second                                          2729192.3395961127
[2026-01-12 18:57:26] humanoid:converged_worlds                                          8192
[2026-01-12 18:57:26] humanoid:step                                                      364.29383988433983
[2026-01-12 18:57:26] humanoid:step.forward                                              361.76275029720273
[2026-01-12 18:57:26] humanoid:step.forward.fwd_position                                 89.69937137590023
[2026-01-12 18:57:26] humanoid:step.forward.fwd_position.kinematics                      16.32935900670418
...
```

## Configuration

Benchmarks are defined in `config.txt`, a simple text file with space-separated columns. Each line specifies a benchmark with the following format:

```
NAME MJCF NWORLD NCONMAX NJMAX NSTEP REPLAY
```

Where:
- `NAME`: Benchmark identifier (used for filtering and in output)
- `MJCF`: Path to the MJCF model file (relative to the `benchmarks` directory)
- `NWORLD`: Number of parallel simulations to run
- `NCONMAX`: Maximum number of contacts per world
- `NJMAX`: Maximum number of constraints per world
- `NSTEP`: Number of simulation steps (use `-` for default of 1000)
- `REPLAY`: Keyframe sequence prefix to replay (use `-` if not needed)

Example configuration:

```
# Format: NAME MJCF NWORLD NCONMAX NJMAX NSTEP REPLAY
humanoid                  humanoid/humanoid.xml                8192  24   64   -     -
aloha_pot                 aloha_pot/scene.xml                  8192  24   128  -     lift_pot
aloha_cloth               aloha_cloth/scene.xml                32    920  6300 100   -
```

Lines starting with `#` are treated as comments and ignored.

## Adding New Benchmarks

To add a new benchmark:

1. Place your MJCF model in an appropriate subdirectory under `benchmarks/`
2. Add a line to `config.txt` with the appropriate parameters (follow the columnar format)
3. Run `./run.sh -f your_benchmark_name` to test it

## Direct Usage of mjwarp-testspeed

You can also run `mjwarp-testspeed` directly for more control:

```bash
mjwarp-testspeed humanoid/humanoid.xml \
  --nworld=8192 \
  --nconmax=24 \
  --njmax=64 \
  --format=short \
  --event_trace=true \
  --memory=true
```

See `mjwarp-testspeed --help` for all available options.
