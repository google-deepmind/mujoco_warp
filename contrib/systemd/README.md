# Systemd Setup for Nightly Benchmarks

This directory contains systemd user service and timer files for running nightly
MuJoCo Warp benchmark sweeps.  By default, the nightly runs a forward sweep to HEAD,
and pushes the results to the gh-pages branch.  See sweep.py for more details.

## Setup

```bash
# 1. Create directories if they don't exist
mkdir -p ~/.config/systemd/user
mkdir -p ~/.local/share/mjwarp-benchmarks

# 2. Copy the nightly scripts and make sweep.py executable.  sweep.py imports common.py from
#    its own directory, so both files have to be installed together.
cp ../../benchmarks/sweep.py ../../benchmarks/common.py ~/.local/share/mjwarp-benchmarks/
chmod +x ~/.local/share/mjwarp-benchmarks/sweep.py

# 3. Copy the service and timer files
cp mjwarp-nightly.service ~/.config/systemd/user/
cp mjwarp-nightly.timer ~/.config/systemd/user/

# 4. Reload systemd
systemctl --user daemon-reload

# 5. Enable and start the timer
systemctl --user enable mjwarp-nightly.timer
systemctl --user start mjwarp-nightly.timer

# 6. Enable lingering (so timer runs even when you're not logged in)
sudo loginctl enable-linger $USER
```

## Useful Commands

```bash
# Check timer status
systemctl --user status mjwarp-nightly.timer

# List all timers and when they'll run next
systemctl --user list-timers

# Run the benchmark manually (without waiting for timer)
systemctl --user start mjwarp-nightly.service

# View logs
journalctl --user -u mjwarp-nightly.service -f

# View recent logs
journalctl --user -u mjwarp-nightly.service --since "1 hour ago"

# Disable the timer
systemctl --user disable mjwarp-nightly.timer
```

## Configuration

Edit `~/.config/systemd/user/mjwarp-nightly.service` to customize:

- `Environment="CUDA_DEVICE=0"` - Which GPU to use
- `MemoryMax=32G` - Memory limit for the benchmark job

Edit `~/.config/systemd/user/mjwarp-nightly.timer` to change the schedule:

- `OnCalendar=*-*-* 02:00:00` - Default: 2 AM daily
- `OnCalendar=Mon *-*-* 02:00:00` - Example: Mondays only at 2 AM
- `OnCalendar=*-*-* 02,14:00:00` - Example: Twice daily at 2 AM and 2 PM

## Updating

When updating to a newer version of MuJoCo Warp, copy the latest sweep scripts.  Copy both, or
sweep.py will fail at startup on a stale or missing common.py:

```bash
cp /path/to/mujoco_warp/benchmarks/sweep.py /path/to/mujoco_warp/benchmarks/common.py \
  ~/.local/share/mjwarp-benchmarks/
```

If you installed this before sweep.py was split into two files, do the one-time migration below as
well.  Copying only the scripts is not enough: your existing service still runs the old single-file
copy at `~/.local/bin/mjwarp-sweep`, so the nightly would keep benchmarking stale code without ever
reporting an error.

```bash
cp mjwarp-nightly.service ~/.config/systemd/user/
systemctl --user daemon-reload
rm ~/.local/bin/mjwarp-sweep
```

## Acknowledgements

This systemd setup is adapted from [mjlab](https://github.com/mujocolab/mjlab) by
[Kevin Zakka](https://github.com/kevinzakka).
