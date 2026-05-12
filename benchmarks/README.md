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

To execute all benchmarks, from the repository root run:

```bash
uv run python3 benchmarks/run.py
```

This will run all benchmarks and output metrics in a columnar format.

### Filtering Benchmarks

To run specific benchmarks, use the `-f` or `--filter` option with a regex pattern:

```bash
# Run only the humanoid benchmark
uv run python3 benchmarks/run.py -f humanoid

# Run all Apollo variants
uv run python3 benchmarks/run.py -f apollo

# Run all benchmarks with "cloth" in the name
uv run python3 benchmarks/run.py -f cloth
```

## Output Format

The benchmark script outputs metrics in a columnar format:

```
$ uv run python3 benchmarks/run.py -f humanoid
humanoid.jit_duration: 0.3430611090734601
humanoid.run_time 3.0016206190921366
humanoid.steps_per_second 2729192.3395961127
humanoid.converged_worlds 8192
humanoid.step 364.29383988433983
humanoid.step.forward 361.76275029720273
humanoid.step.forward.fwd_position 89.69937137590023
humanoid.step.forward.fwd_position.kinematics 16.32935900670418
...
```

## Configuration

Benchmarks are configured using Python modules. Each benchmark directory (e.g., `benchmarks/humanoid/`) must contain an `__init__.py` file that defines `BENCHMARKS` and optionally `ASSETS`.

### `BENCHMARKS` List

The `BENCHMARKS` list contains dictionaries defining each benchmark variant.

Example from `benchmarks/humanoid/__init__.py`:

```python
BENCHMARKS = [
  {
    "name": "humanoid",
    "mjcf": "humanoid.xml",
    "nworld": 8192,
    "nconmax": 24,
    "njmax": 64,
  },
  {
    "name": "three_humanoids",
    "mjcf": "three_humanoids.xml",
    "nworld": 8192,
    "nconmax": 100,
    "njmax": 192,
  },
]
```

Fields:
- `name`: Unique identifier for the benchmark.
- `mjcf`: Path to the MJCF model file (relative to the benchmark directory).
- `nworld`: Number of parallel rollouts.
- `nconmax`: Maximum number of contacts per world.
- `njmax`: Maximum number of constraints per world.
- `nstep`: (Optional) Number of steps per rollout.
- `replay`: (Optional) NPZ file with ctrl sequence to replay.
- `assets`: (Optional) List of asset mappings (see below).

### `ASSETS` List

If a benchmark requires external assets (e.g., from MuJoCo Menagerie), they can be defined in the `ASSETS` list. `run.py` will automatically fetch them before running the benchmark.

Example:

```python
ASSETS = [
  {
    "source": "https://github.com/google-deepmind/mujoco_menagerie.git",
    "ref": "affef0836947b64cc06c4ab1cbf0152835693374",
  }
]
```

Each benchmark's `assets` field is a list of tuples mapping fetched assets into the benchmark workspace:

```python
"assets": [
  (ASSETS[0], "aloha"),                                        # copy aloha/ directory
  (ASSETS[1], "model/plugin/sdf/asset", "assets"),              # copy with rename
  (ASSETS[2], "aloha_sim/assets/ycb/*/google_64k", "assets/ycb"),  # glob pattern
]
```

Each tuple is `(asset, src_subpath[, dst_subpath])`. If `src_subpath` contains a `*`, it is expanded as a glob and each matching directory is copied into `dst_subpath` using the `*`-matched segment as the subdirectory name.

## Adding New Benchmarks

To add a new benchmark:

1. Create a new directory under `benchmarks/` (e.g., `benchmarks/my_robot/`).
2. Place your MJCF model and any local assets in that directory.
3. Create an `__init__.py` file in that directory defining `BENCHMARKS` (and `ASSETS` if needed).
4. Run `uv run python3 benchmarks/run.py -f my_robot` to test it.

## Direct Usage of mjwarp-testspeed

You can also run `mjwarp-testspeed` directly for more control:

```bash
uv run mjwarp-testspeed benchmarks/humanoid/humanoid.xml \
  --nworld=8192 \
  --nconmax=24 \
  --njmax=64 \
  --format=short \
  --event_trace=true
```

See `mjwarp-testspeed --help` for all available options.
