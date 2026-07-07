#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import ts from "typescript";

const ignoredDirectories = new Set([
  ".git",
  "coverage",
  "dist",
  "node_modules",
  "target",
]);
const sourceExtensions = new Set([".ts", ".tsx", ".mts", ".cts"]);

function sourceFiles(directory) {
  const files = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    if (entry.isDirectory() && ignoredDirectories.has(entry.name)) continue;
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...sourceFiles(fullPath));
    else if (entry.isFile() && sourceExtensions.has(path.extname(entry.name))) files.push(fullPath);
  }
  return files;
}

let failures = 0;
for (const file of sourceFiles(process.cwd())) {
  const source = fs.readFileSync(file, "utf8");
  const result = ts.transpileModule(source, {
    fileName: file,
    reportDiagnostics: true,
    compilerOptions: {
      jsx: ts.JsxEmit.Preserve,
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
  });
  for (const diagnostic of result.diagnostics ?? []) {
    if (diagnostic.category !== ts.DiagnosticCategory.Error) continue;
    failures += 1;
    const message = ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n");
    const position = diagnostic.start == null
      ? ""
      : (() => {
          const sourceFile = ts.createSourceFile(file, source, ts.ScriptTarget.ES2022, true);
          const point = sourceFile.getLineAndCharacterOfPosition(diagnostic.start);
          return `:${point.line + 1}:${point.character + 1}`;
        })();
    console.error(`${path.relative(process.cwd(), file)}${position}: ${message}`);
  }
}

process.exitCode = failures === 0 ? 0 : 1;
