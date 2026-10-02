# Contributing to MuJoCo Warp

## Contributor License Agreement & Rules

- Contributions must be accompanied by a [Contributor License Agreement](https://cla.developers.google.com/) (CLA).
- If you sign the CLA after opening a PR, comment `@google-cla recheck` on the PR to trigger a status update.
- **No AI co-authors:** All commit authors must have signed the CLA. Do not add AI assistants/agents as commit authors
  or co-authors; commits with unsigned co-authors fail CLA checks.
- Push branches to your own fork, not directly to `google-deepmind/mujoco_warp`.

## Development Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e .[dev]
uv run pre-commit install
```

- Always use `uv run`, not plain `python`.
- Prefer running individual tests (`uv run pytest path/to/test.py -k test_name`) over the full test suite during iteration.
- Run `uv run pytest -n 8` before creating a pull request.

## Code Style & Warp GPU Conventions

- Line length limit is 128 characters. Docstring length limit is 100 characters.
- All new source files must begin with the Apache 2.0 copyright header (`# Copyright 2026 The Newton Developers`).
- **Pre-commit & kernel analyzer:** Run `uv run pre-commit run --all-files`. For GPU kernels, follow parameter conventions
  (`_in` for inputs, `_out` for outputs, matching `Model`/`Data` field order) and avoid `# kernel_analyzer: off` blocks.
- **Global memory traffic:** Reads and writes to device global memory (`wp.array`) are expensive:
  - Hoist global array reads (`m.*`, `d.*`) into local variables (registers) before loops; minimize global memory writes.
  - Hoist loop-invariant calculations and precompute reciprocals outside loops (e.g. `inv_h = 1.0 / float(h)`).
- **Launch guards:** Do not guard kernel launches with dimension checks (e.g. `if na > 0: wp.launch(...)`) when the quantity is
  already part of the launch dimension. Launching with dimension 0 is a safe no-op in Warp.
- **Operator precedence:** Disambiguate bitwise and logical expressions with parentheses (e.g. in `wp.static` conditions).
- **Quaternion convention:** All public APIs, `Model`, and `Data` fields follow the MuJoCo convention `(w, x, y, z)`. Explicitly
  convert or document when passing to native Warp functions (`wp.transform`).
- **CUDA graph compatibility:** Avoid CPU-GPU synchronization (`.numpy()`, `.item()`, D2H copies) during simulation steps.

## Writing Tests

- **Test coverage for fixes and features:** PRs fixing a bug or adding a feature should generally include at least one test that
  fails on main (before the change) and passes with the PR. For bug fixes, this reproduces the issue; for new features, this
  verifies the added functionality.
- **Inline XML in `test_data.fixture`:** Always use `test_data.fixture(xml="""...""")` with the XML string literal passed
  directly. Do not assign an intermediate `xml = ...` variable or use standalone `.xml` files unless testing multi-asset scenes.
- **Pre-fill tested fields:** Because `test_data.fixture` initializes `d` with MuJoCo CPU values, reset tested output fields to
  `wp.inf` (floats) or `-1` (ints) before calling the function under test (e.g. `d.sensordata.fill_(wp.inf)`).
- **Multi-world batching (`nworld: 1, 2`):** Parameterize new tests with `@parameterized.parameters(1, 2)`. When `nworld == 2`,
  perturb world 1 so the worlds are heterogeneous, evaluate per-world CPU references, and assert world 0 and world 1 outputs are
  not identical.
- **Parity against CPU:** Compare outputs against upstream CPU reference by invoking corresponding `mujoco.mj_*` functions
  (`mj_forward`, `mj_sensorPos`, `mjd_smooth_vel`) using standard tolerances (`rtol=1e-5, atol=1e-5`).
- **Non-degenerate scenes:** Ensure test scenes include non-zero velocities (`fixture(qvel_noise=...)`) or rotations when
  testing velocity-dependent, spatial, or gyroscopic terms.
- **Rendering tests:** Assert analytical geometric invariants (camera ray projection math, bounds) rather than brittle
  pixel-level comparisons.
- **Dependencies:** Avoid `mock`, `SimpleNamespace`, or repo directory traversals (`__file__.parents`). Use `etils.epath` where
  path handling is needed.

## Contributing Benchmarks

Benchmarks live under `benchmarks/<name>/`:

```
benchmarks/<name>/
├── __init__.py           # Defines BENCHMARKS and optional ASSETS lists
├── <model>.xml           # MJCF XML model file
├── <trajectory>.npz      # (Optional) Control trajectory replay sequence
├── rollout_<name>.webp   # Visual animation of benchmark rollout
└── README.md             # Overview, model properties table, and rollout preview
```

- **Configuration:** Define `BENCHMARKS = [{...}]` in `__init__.py` with `name`, `mjcf`, `nworld` (typically 2048–8192),
  `nconmax`, and `njmax` (and optional `nccdmax`, `nvmax`, `replay`, `assets`).
- **Rendering & sleeping:** For vision benchmarks, set `"function": "render"` and define camera parameters. For sleeping
  benchmarks, set `override="opt.enableflags=SLEEP"` and `init_asleep=True`.
- **Zero buffer overflow:** Benchmarks must run with zero overflow (`warn_overflow = 0` and `converged_worlds == nworld`).
  Overflows drop contacts/constraints and distort throughput numbers.
- **External assets:** Large meshes and assets must be fetched via shallow git clones defined in `ASSETS` and mapped in `assets`.
  Never commit large asset packs directly to git.
- **Deterministic replay:** Dynamic benchmarks should supply an `.npz` control replay trajectory.
- **Documentation:** Provide a `README.md` with scenario description, property table (Bodies, DOFs, Actuators, Geoms, Timestep,
  Solver, Integrator, Matrix Format), and an embedded `.webp` rollout animation.
- **Verification:** Run `uv run python3 benchmarks/run.py -f <name>` (headless) and with `--view` (interactive viewer; note
  `--view` is not supported for `function="render"`).

## Pull Requests & Code Review

- **Commit history:** Amending commits is fine before review starts. Once a PR is under review, use new commits.
- **Performance evidence:** Include before-and-after throughput benchmarks (`benchmarks/run.py` or `mjwarp-testspeed`) with
  identical `nworld` and zero overflow for changes to simulation or rendering hot paths.
- **Responding to reviews:** Reply to each comment confirming what was changed (or why it wasn't), resolve addressed threads,
  and leave a summary comment on the PR when done.
- **Pass pre-commit and tests:** Ensure all pre-commit hooks (`uv run pre-commit run --all-files`) and relevant unit tests
  pass cleanly before requesting review.
- **Cleanliness:** Remove temporary debug scripts, test files, and unused imports before requesting review.

## Community Guidelines

This project follows [Google's Open Source Community Guidelines](https://opensource.google/conduct/).
