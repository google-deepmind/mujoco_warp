# Systemd Setup for Nightly Benchmarks

This directory contains systemd user service and timer files for running nightly
MuJoCo Warp benchmarks.

## Setup

```bash
# 1. Create directories if they don't exist
mkdir -p ~/.config/systemd/user
mkdir -p ~/.local/bin

# 2. Copy the nightly script and make it executable
cp ../../benchmarks/nightly.sh ~/.local/bin/mjwarp-nightly
chmod +x ~/.local/bin/mjwarp-nightly

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

When updating to a newer version of MuJoCo Warp, copy the latest nightly script:

```bash
cp /path/to/mujoco_warp/benchmarks/nightly.sh ~/.local/bin/mjwarp-nightly
```

## Acknowledgements

This systemd setup is adapted from [mjlab](https://github.com/mujocolab/mjlab) by
[Kevin Zakka](https://github.com/kevinzakka).
