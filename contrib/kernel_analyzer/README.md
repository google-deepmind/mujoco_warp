# Kernel Analyzer

Kernel Analyzer checks warp kernels to ensure correctness and conformity.  It comes with both a CLI (which can be used within github CI) and also a vscode plugin for automatic kernel issue highlighting.

# CLI usage

```bash
python contrib/kernel_analyzer/kernel_analyzer/cli.py --files somefile.py --types mujoco_warp/_src/types.py 
```

# CLI for github CI

```bash
python contrib/kernel_analyzer/kernel_analyzer/cli.py --files somefile.py --types mujoco_warp/_src/types.py 
```

# VSCode plugin

Enjoy kernel analysis directly within vscode.

## Installing kernel analyzer

1. Create a new python env (`python3 -m venv env`) or use your existing mjwarp env (`source env/bin/activate`).
2. Within the python env, install the kernel analyzer python dependencies:
  ```bash
  pip install -r contrib/kernel_analyzer/kernel_analyzer/requirements.txt
  ```
3. Inside vscode, navigate to `contrib/kernel_analyzer/`
4. Right click on `kernel-analyzer-{version}.vsix` file
5. Select "Install Extension VSIX"
6. Open vscode settings and navigate to `Extensions > Kernel Analyzer`
7. Set **Python Path** to the `bin/python` of the env you set up in step 1, e.g. `/home/$USER/work/mujoco_warp/env/bin/python`
8. Set **Types Path** to the location of `types.py` in your checked out code, e.g. `/home/$USER/work/mujoco_warp/mujoco_warp/_src/types.py`

## Plugin Development

Create a debug configuration in `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "args": [
        "--extensionDevelopmentPath=${workspaceFolder}/contrib/kernel_analyzer"
      ],
      "name": "Launch Extension",
      "outFiles": [
        "${workspaceFolder}/contrib/kernel_analyzer/out/**/*.js"
      ],
      "preLaunchTask": "${defaultBuildTask}",
      "request": "launch",
      "type": "extensionHost",
    }
  ]
}
```

# Packaging a new vscode plugin

```bash
npm run package
```