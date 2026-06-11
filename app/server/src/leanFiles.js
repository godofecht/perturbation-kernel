// Discovery + heuristic parsing of the Lean sources under LEAN_PROJECT_DIR.
//
// We intentionally keep this a lightweight, regex-driven structural parser
// rather than anything that needs Lean itself: it powers the file browser and
// the proof-structure outline in the PWA, which must work even when no Lean
// toolchain is installed (visualise-only mode). Live goal states come from the
// REPL (see leanRepl.js); this file only describes shape.

import { promises as fs } from 'node:fs';
import path from 'node:path';

// Top-level declaration keywords we surface in the outline. Order matters only
// for display grouping; detection is keyword-based.
const DECL_KEYWORDS = [
  'theorem',
  'lemma',
  'def',
  'abbrev',
  'structure',
  'inductive',
  'instance',
  'example',
  'class',
];

// Modifiers that may precede a declaration keyword on the same line.
const MODIFIERS = new Set([
  'private',
  'protected',
  'noncomputable',
  'partial',
  'unsafe',
  'scoped',
  'local',
  '@[simp]',
]);

const SKIP_DIRS = new Set(['.lake', 'lake-packages', '.git', 'build']);

/** Recursively collect *.lean files under `root`, returning repo-relative posix paths. */
export async function listLeanFiles(root) {
  const out = [];
  async function walk(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const e of entries) {
      if (e.isDirectory()) {
        if (SKIP_DIRS.has(e.name)) continue;
        await walk(path.join(dir, e.name));
      } else if (e.isFile() && e.name.endsWith('.lean')) {
        out.push(path.join(dir, e.name));
      }
    }
  }
  await walk(root);
  return out.sort();
}

/**
 * Strip a leading `@[...]` attribute and any modifier tokens, returning the
 * remaining tokens of a declaration-opening line. Returns null when the line
 * does not open a declaration.
 */
function declTokens(line) {
  const trimmed = line.trimStart();
  if (trimmed.startsWith('--') || trimmed.startsWith('/-')) return null;
  // Drop a leading attribute block like `@[simp, foo]`.
  const withoutAttr = trimmed.replace(/^@\[[^\]]*\]\s*/, '');
  const tokens = withoutAttr.split(/\s+/).filter(Boolean);
  let i = 0;
  while (i < tokens.length && MODIFIERS.has(tokens[i])) i += 1;
  if (i >= tokens.length) return null;
  if (!DECL_KEYWORDS.includes(tokens[i])) return null;
  return tokens.slice(i);
}

/** Pull a declaration name out of the tokens after its keyword. */
function declName(tokens) {
  const kind = tokens[0];
  if (kind === 'example') return 'example';
  const raw = tokens[1] ?? '';
  // Names can be followed by `:` or `(` with no space; cut at the first such.
  const name = raw.split(/[:({\[]/)[0];
  // Anonymous instances (`instance : Foo := …`) and the like have no name.
  if (!name) return kind === 'instance' ? 'instance' : `(${kind})`;
  return name;
}

/**
 * Parse a Lean source into a flat list of top-level declarations with:
 *   { kind, name, startLine, endLine, hasSorry, tactics }
 * Lines are 1-based. `tactics` is a best-effort, indentation-derived outline of
 * the proof body (the tokens that open each tactic line inside a `by` block).
 */
export function parseDeclarations(source) {
  const lines = source.split('\n');
  const starts = [];
  for (let n = 0; n < lines.length; n += 1) {
    // Only treat column-0 (or near-0) lines as top-level openers; deeper
    // indentation is part of a proof/term body, not a new declaration.
    const indent = lines[n].length - lines[n].trimStart().length;
    if (indent > 0) continue;
    const tokens = declTokens(lines[n]);
    if (tokens) starts.push({ line: n, kind: tokens[0], name: declName(tokens) });
  }

  const decls = [];
  for (let k = 0; k < starts.length; k += 1) {
    const startLine = starts[k].line;
    const endLine = k + 1 < starts.length ? starts[k + 1].line - 1 : lines.length - 1;
    const body = lines.slice(startLine, endLine + 1);
    const text = body.join('\n');
    decls.push({
      kind: starts[k].kind,
      name: starts[k].name,
      startLine: startLine + 1,
      endLine: endLine + 1,
      hasSorry: /\bsorry\b/.test(text),
      tactics: tacticOutline(body),
    });
  }
  return decls;
}

/**
 * Heuristic tactic outline: find a `by` and collect the first token of each
 * subsequent indented line, deduplicated against trivial noise. This is for
 * human-readable structure in the UI, not a faithful proof tree.
 */
function tacticOutline(bodyLines) {
  const joined = bodyLines.join('\n');
  const byIdx = joined.search(/(^|\s):=\s*by\b|(^|\s)\bby\b/);
  if (byIdx === -1) return [];
  // Work line-wise from the line containing `by`.
  let started = false;
  const tactics = [];
  for (const raw of bodyLines) {
    const line = raw.trim();
    if (!started) {
      if (/\bby\b\s*$/.test(raw) || /:=\s*by\b/.test(raw)) started = true;
      // a `by tac` on the same line: capture the trailing tactic too
      const inline = raw.match(/\bby\s+(\S+)/);
      if (inline && !/\bby\s*$/.test(raw)) {
        started = true;
        tactics.push(inline[1].replace(/[;,]$/, ''));
      }
      continue;
    }
    if (!line || line.startsWith('--')) continue;
    const head = line.split(/[\s({\[]/)[0].replace(/[;,]$/, '');
    if (head && head !== '·' && head !== '|') tactics.push(head);
  }
  return tactics;
}

/** Build the browser manifest: every file with its declaration summary. */
export async function buildManifest(root) {
  const files = await listLeanFiles(root);
  const result = [];
  for (const abs of files) {
    const source = await fs.readFile(abs, 'utf8');
    const decls = parseDeclarations(source);
    result.push({
      path: path.relative(root, abs).split(path.sep).join('/'),
      lines: source.split('\n').length,
      declarations: decls,
      sorryCount: decls.filter((d) => d.hasSorry).length,
    });
  }
  return result;
}

/** Read one file's source + parsed declarations. `rel` is project-relative. */
export async function readLeanFile(root, rel) {
  // Guard against path traversal: resolved path must stay under root.
  const abs = path.resolve(root, rel);
  const normRoot = path.resolve(root);
  if (abs !== normRoot && !abs.startsWith(normRoot + path.sep)) {
    throw new Error('path escapes project root');
  }
  const source = await fs.readFile(abs, 'utf8');
  return { path: rel, content: source, declarations: parseDeclarations(source) };
}
