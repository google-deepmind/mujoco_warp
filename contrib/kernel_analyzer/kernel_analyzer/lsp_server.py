# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
from pathlib import Path
from typing import Any, Dict, List

import ast_analyzer
from lsprotocol.types import ConfigurationItem
from lsprotocol.types import Diagnostic
from lsprotocol.types import DiagnosticSeverity
from lsprotocol.types import Position
from lsprotocol.types import Range
from lsprotocol.types import WorkspaceConfigurationParams
from pygls.server import LanguageServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class KernelAnalyzerLanguageServer(LanguageServer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.issues: Dict[str, List[Any]] = {}


_server = KernelAnalyzerLanguageServer("kernel-analyzer", "v0.1")


@_server.feature("textDocument/didOpen")
@_server.feature("textDocument/didChange")
@_server.feature("textDocument/didSave")
async def validate(ls: KernelAnalyzerLanguageServer, params):
  """Validate the document using core_logic.ast_analyzer."""
  config = await ls.get_configuration_async(
    WorkspaceConfigurationParams(
      items=[ConfigurationItem(scope_uri="", section="kernelAnalyzer.typesPath")]
    )
  )
  type_source = Path(config[0]).read_text()

  text_doc = ls.workspace.get_text_document(params.text_document.uri)
  source = text_doc.source
  diagnostics: List[Diagnostic] = []
  ls.issues[text_doc.uri] = []
  logging.info(f"Validating document: {text_doc.uri}")

  try:
    issues = ast_analyzer.analyze(source, text_doc.uri, type_source)
    logging.info(f"Analyzer found {len(issues)} issues in {text_doc.uri}")
    # store for potential future use (code actions)
    ls.issues[text_doc.uri] = issues

    for issue in issues:
      diag = Diagnostic(
        range=Range(
          start=Position(line=issue.node.lineno - 1, character=issue.node.col_offset),
          end=Position(
            line=issue.node.end_lineno - 1, character=issue.node.end_col_offset
          ),
        ),
        message=str(issue),
        severity=DiagnosticSeverity.Warning,  # Yellow underline
        code=type(issue).__name__,
        source="Kernel Analyzer",
      )
      diagnostics.append(diag)

  except Exception as e:
    # Log errors during validation
    logging.error(f"Error during validation for {text_doc.uri}: {e}", exc_info=True)
    diagnostics.append(
      Diagnostic(
        range=Range(
          start=Position(line=0, character=0), end=Position(line=0, character=0)
        ),
        message=str(e),
      )
    )
    # Optionally add a single general error diagnostic to the file start
    # diagnostics.append(Diagnostic(...))

  logging.info(f"Publishing {len(diagnostics)} diagnostics for {text_doc.uri}")
  ls.publish_diagnostics(text_doc.uri, diagnostics)


if __name__ == "__main__":
  logging.info("Starting Parameter Sorter Language Server...")
  _server.start_io()
