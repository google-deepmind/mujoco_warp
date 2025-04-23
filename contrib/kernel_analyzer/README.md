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

1. Inside vscode, navigate to `contrib/kernel_analyzer/`
2. Right click on `kernel-analyzer-{version}.vsix` file
3. Select "Install Extension VSIX"

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