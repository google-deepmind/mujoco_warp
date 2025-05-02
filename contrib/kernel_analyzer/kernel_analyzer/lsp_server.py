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
from typing import Any, Dict, List, Optional

import ast_analyzer
from lsprotocol import types
from lsprotocol.types import InitializeResult
from lsprotocol.types import TextDocumentSyncOptions
from lsprotocol.types import InitializeParams
from lsprotocol.types import ServerCapabilities
from lsprotocol.types import TextDocumentSyncKind
from lsprotocol.types import ConfigurationItem
from lsprotocol.types import Diagnostic
from lsprotocol.types import InitializeResultServerInfoType
from lsprotocol.types import DiagnosticOptions
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


_server = KernelAnalyzerLanguageServer("kernel-analyzer", "v0.2")


@_server.feature(types.INITIALIZE)
async def initialize(client: LanguageServer, params: InitializeParams):
  return InitializeResult(
    capabilities=ServerCapabilities(
      text_document_sync=TextDocumentSyncOptions(
        open_close=True,
        change=TextDocumentSyncKind.Full,
        save=True,
      ),
      diagnostic_provider=DiagnosticOptions(
        inter_file_dependencies=False,
        workspace_diagnostics=False,
      ),
      code_action_provider=True,
    ),
    server_info=InitializeResultServerInfoType(
      name="kernel-analyzer",
      version="v0.2",
    ),
  )


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
        data=issue.__dict__,
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


@_server.feature(types.TEXT_DOCUMENT_CODE_ACTION)
async def code_action(ls: KernelAnalyzerLanguageServer, params: types.CodeActionParams):
  uri = params.text_document.uri
  diagnostics = params.context.diagnostics
  code_actions: List[types.CodeAction] = []
  document = ls.workspace.get_text_document(uri)

  for diagnostic in diagnostics:
    if diagnostic.code == "TypeMismatch":
      expected_type = diagnostic.data["expected_type"]
      annotation_range = diagnostic.range

      # Adjust the start of the range to be after the colon and any whitespace
      if document:
        line_text = document.lines[annotation_range.start.line]
        colon_index = line_text.find(":", annotation_range.start.character)
        if colon_index != -1:
          annotation_start = types.Position(
            line=annotation_range.start.line, character=colon_index + 1
          )
          while (
            annotation_start.character < len(line_text)
            and line_text[annotation_start.character].isspace()
          ):
            annotation_start.character += 1
          annotation_range = types.Range(
            start=annotation_start, end=annotation_range.end
          )

      replacement_text = expected_type
      edit = types.WorkspaceEdit(
        changes={
          uri: [
            types.TextEdit(
              range=annotation_range,
              new_text=replacement_text,
            )
          ]
        }
      )

      code_action = types.CodeAction(
        title=f"Change type to '{expected_type}'",
        kind=types.CodeActionKind.QuickFix,
        diagnostics=[diagnostic],
        edit=edit,
      )
      code_actions.append(code_action)
    if diagnostic.code == "InvalidParamOrder":
      new_text: str = diagnostic.data["expected_full"]
      ar = diagnostic.data["arg_range"]
      arg_range = types.Range(
        start=types.Position(line=ar[0][0], character=ar[0][1]),
        end=types.Position(line=ar[1][0], character=ar[1][1]),
      )

      if document:
        new_text = "\n".join(["  " + q for q in new_text])
        new_text = "\n" + new_text + "\n"

        edit = types.WorkspaceEdit(
          changes={
            uri: [
              types.TextEdit(
                range=arg_range,
                new_text=new_text,
              )
            ]
          }
        )
        code_action = types.CodeAction(
          title="Fix Parameter Order",
          kind=types.CodeActionKind.QuickFix,
          diagnostics=[diagnostic],
          edit=edit,
        )
        code_actions.append(code_action)

  return code_actions


if __name__ == "__main__":
  logging.info("Starting Parameter Sorter Language Server...")
  _server.start_io()
