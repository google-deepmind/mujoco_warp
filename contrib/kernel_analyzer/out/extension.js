"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const path = __importStar(require("path"));
const vscode_1 = require("vscode");
const node_1 = require("vscode-languageclient/node");
let client; // Reference to the language client
let outputChannel; // Channel for logging
// This function is called when your extension is activated
// Activation happens based on 'activationEvents' in the main package.json
function activate(context) {
    // Create an output channel for logging LSP communication / issues
    outputChannel = vscode_1.window.createOutputChannel("Kernel Analyzer LSP");
    outputChannel.appendLine('Activating Kernel Analyzer extension...');
    console.log('Kernel Analyzer Extension Client activating...');
    let pythonPath = vscode_1.workspace.getConfiguration('kernelAnalyzer').get('pythonPath');
    if (!pythonPath) {
        pythonPath = vscode_1.workspace.getConfiguration('python').get('defaultInterpreterPath');
    }
    if (!pythonPath) {
        vscode_1.window.showErrorMessage('Python interpreter path not found. Please configure the Python extension (`python.defaultInterpreterPath`) or ensure Python is in your PATH.');
        outputChannel.appendLine('Python interpreter path not configured.');
        return; // Stop activation if Python isn't found
    }
    outputChannel.appendLine(`Using Python interpreter: ${pythonPath}`);
    const serverModule = context.asAbsolutePath(path.join('kernel_analyzer', 'lsp_server.py'));
    outputChannel.appendLine(`LSP Server script path: ${serverModule}`);
    const serverOptions = {
        command: pythonPath, // The Python executable
        args: [serverModule, /* '-v' */], // The script to run (add '-v' etc. if your server accepts args)
        transport: node_1.TransportKind.stdio // Communicate via standard input/output
    };
    const clientOptions = {
        // Register the server for 'python' documents
        documentSelector: [{ scheme: 'file', language: 'python' }],
        outputChannel: outputChannel // Use the dedicated channel for LSP logs
    };
    outputChannel.appendLine('Creating Language Client...');
    client = new node_1.LanguageClient('kernelAnalyzerLsp', // Unique ID for the client
    'Kernel Analyzer LSP', // Name shown in Output channel selector
    serverOptions, clientOptions);
    // Start the client. This will also launch the server process.
    outputChannel.appendLine('Starting Language Client...');
    client.start()
        .then(() => {
        outputChannel.appendLine('Kernel Analyzer Language Client started successfully.');
        console.log('Kernel Analyzer Language Client started.');
    })
        .catch((error) => {
        vscode_1.window.showErrorMessage(`Failed to start Kernel Analyzer Language Client: ${error}. See Output channel for details.`);
        outputChannel.appendLine(`Failed to start Language Client: ${error}`);
        console.error('Failed to start Kernel Analyzer Language Client:', error);
    });
}
// This function is called when your extension is deactivated
function deactivate() {
    outputChannel?.appendLine('Deactivating Kernel Analyzer extension...');
    if (!client) {
        return undefined;
    }
    // Stop the language client (which also stops the server process)
    return client.stop();
}
//# sourceMappingURL=extension.js.map