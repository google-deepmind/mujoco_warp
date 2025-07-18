# MuJoCo Warp Benchmark Suite

MJWarp uses [airspeed velocity](https://github.com/airspeed-velocity/asv) for benchmarks.

Make sure you install MJWarp in develop so you can run the `asv` command:

```
pip install -e .[dev,cuda]
```

Run benchmarks like so at the top level of the `mujoco_warp` checkout:

```
asv run
```

You should see output that looks like this:

```
Couldn't load asv.plugins._mamba_helpers because
No module named 'libmambapy'
· Creating environments
· Discovering benchmarks
·· Uninstalling from virtualenv-py3.12
·· Installing c6b97641 <asv> into virtualenv-py3.12.
· Running 7 total benchmarks (1 commits * 1 environments * 7 benchmarks)
[ 0.00%] · For mujoco-warp commit c6b97641 <asv>:
[ 0.00%] ·· Benchmarking virtualenv-py3.12
[14.29%] ··· benchmark.AlohaPot.time_function
[14.29%] ··· ================== ============
                  function
             ------------------ ------------
                 collision        365±10ns
                fwd_position      443±10ns
                fwd_velocity     31.2±0.2ns
               fwd_actuation     13.3±0.2ns
              fwd_acceleration   8.58±0.2ns
                   solve         612±200ns
                    step          1.90±1μs
             ================== ============
...
```

Benchmarks are slow to run - if you would like to quickly verify benchmarks are working, you can run with `-q`:

```
asv run -q
```

You can also benchmark your own branch:

```
asv run $MYBRANCH
```

See the [airspeed velocity documentation](https://asv.readthedocs.io/en/latest/index.html) for more information.
