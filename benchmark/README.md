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
· Running 6 total benchmarks (1 commits * 1 environments * 6 benchmarks)
[ 0.00%] · For mujoco-warp commit 603429ca <asv-2>:
[ 0.00%] ·· Benchmarking virtualenv-py3.12
[16.67%] ··· Setting up benchmark:85                                                                                                                                   ok
[16.67%] ··· benchmark.ApptronikApolloFlat.track_metric                                                                                                                ok
[16.67%] ··· =============================================== =====================
                                 function                                         
             ----------------------------------------------- ---------------------
                               jit_duration                   0.21160659193992615 
                            solver_niter_mean                  3.263658447265625  
                             solver_niter_p95                         5.0         
                         device_memory_allocated                   887095296      
                                   step                        758.2121635787189  
                               step.forward                    753.6720001371577  
                        step.forward.fwd_position              106.35671483032638 
                   step.forward.fwd_position.kinematics        38.16051127068931  
                    step.forward.fwd_position.com_pos          11.754250321246218 
                    step.forward.fwd_position.camlight         2.2997500059318554 
                      step.forward.fwd_position.crb             18.2445002192253  
                step.forward.fwd_position.tendon_armature     0.17412499956037664 
                   step.forward.fwd_position.collision          6.07399994669322  
...
```

Benchmarks are slow to run - if you would like to filter a single benchmark, use `-b`:

```
asv run -b ApptronikApolloFlat
```

You can also benchmark your own branch:

```
asv run $MYBRANCH
```

See the [airspeed velocity documentation](https://asv.readthedocs.io/en/latest/index.html) for more information.
