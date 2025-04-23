import * as path from 'path';
import { workspace, ExtensionContext, window, OutputChannel } from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient; // Reference to the language client
let outputChannel: OutputChannel; // Channel for logging

// This function is called when your extension is activated
// Activation happens based on 'activationEvents' in the main package.json
export function activate(context: ExtensionContext) {

    // Create an output channel for logging LSP communication / issues
    outputChannel = window.createOutputChannel("Kernel Analyzer LSP");
    outputChannel.appendLine('Activating Kernel Analyzer extension...');
    console.log('Kernel Analyzer Extension Client activating...');

    let pythonPath = workspace.getConfiguration('kernelAnalyzer').get<string>('pythonPath');

    if (!pythonPath) {
        pythonPath = workspace.getConfiguration('python').get<string>('defaultInterpreterPath');
    }

    if (!pythonPath) {
        window.showErrorMessage(
            'Python interpreter path not found. Please configure the Python extension (`python.defaultInterpreterPath`) or ensure Python is in your PATH.'
        );
        outputChannel.appendLine('Python interpreter path not configured.');
        return; // Stop activation if Python isn't found
    }
    outputChannel.appendLine(`Using Python interpreter: ${pythonPath}`);

    const serverModule = context.asAbsolutePath(
        path.join('kernel_analyzer', 'lsp_server.py')
    );
    outputChannel.appendLine(`LSP Server script path: ${serverModule}`);

    const serverOptions: ServerOptions = {
        command: pythonPath,                // The Python executable
        args: [serverModule, /* '-v' */],   // The script to run (add '-v' etc. if your server accepts args)
        transport: TransportKind.stdio      // Communicate via standard input/output
    };

    const clientOptions: LanguageClientOptions = {
        // Register the server for 'python' documents
        documentSelector: [{ scheme: 'file', language: 'python' }],
        outputChannel: outputChannel // Use the dedicated channel for LSP logs
    };

    outputChannel.appendLine('Creating Language Client...');
    client = new LanguageClient(
        'kernelAnalyzerLsp',      // Unique ID for the client
        'Kernel Analyzer LSP',    // Name shown in Output channel selector
        serverOptions,
        clientOptions
    );

    // Start the client. This will also launch the server process.
    outputChannel.appendLine('Starting Language Client...');
    client.start()
        .then(() => {
            outputChannel.appendLine('Kernel Analyzer Language Client started successfully.');
            console.log('Kernel Analyzer Language Client started.');
        })
        .catch((error) => {
            window.showErrorMessage(`Failed to start Kernel Analyzer Language Client: ${error}. See Output channel for details.`);
            outputChannel.appendLine(`Failed to start Language Client: ${error}`);
            console.error('Failed to start Kernel Analyzer Language Client:', error);
        });
}

// This function is called when your extension is deactivated
export function deactivate(): Thenable<void> | undefined {
    outputChannel?.appendLine('Deactivating Kernel Analyzer extension...');
    if (!client) {
        return undefined;
    }
    // Stop the language client (which also stops the server process)
    return client.stop();
}
