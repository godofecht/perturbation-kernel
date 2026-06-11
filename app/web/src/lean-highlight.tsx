// Minimal Lean 4 syntax highlighter. Dependency-free on purpose: a full
// TextMate grammar (Shiki) would bloat the bundle and hurt offline load on a
// phone. This is a pragmatic tokenizer good enough for reading proofs.

import { useMemo } from 'react';

type TokType =
  | 'comment'
  | 'string'
  | 'keyword'
  | 'tactic'
  | 'sorry'
  | 'number'
  | 'op'
  | 'attr'
  | 'text';

interface Tok {
  type: TokType;
  value: string;
}

const KEYWORDS = new Set([
  'theorem', 'lemma', 'def', 'abbrev', 'example', 'structure', 'inductive',
  'instance', 'class', 'where', 'extends', 'deriving', 'with', 'fun', 'let',
  'in', 'if', 'then', 'else', 'match', 'do', 'return', 'namespace', 'end',
  'open', 'import', 'variable', 'variables', 'universe', 'section', 'mutual',
  'noncomputable', 'partial', 'unsafe', 'private', 'protected', 'attribute',
  'set_option', 'scoped', 'local', 'macro', 'macro_rules', 'notation',
  'syntax', 'elab', 'by', 'from', 'have', 'show', 'suffices', 'calc',
  'Type', 'Prop', 'Sort',
]);

// Common Mathlib/Lean tactics — highlighted distinctly so proof scripts read
// well. Not exhaustive; unknown identifiers fall through to plain text.
const TACTICS = new Set([
  'intro', 'intros', 'exact', 'apply', 'refine', 'rw', 'rwa', 'simp', 'simp_all',
  'simpa', 'dsimp', 'ring', 'ring_nf', 'linarith', 'nlinarith', 'omega', 'norm_num',
  'constructor', 'cases', 'rcases', 'obtain', 'rintro', 'induction', 'use', 'exists',
  'ext', 'funext', 'congr', 'trivial', 'rfl', 'assumption', 'contradiction',
  'unfold', 'change', 'convert', 'calc', 'gcongr', 'positivity', 'field_simp',
  'push_cast', 'norm_cast', 'measurability', 'fun_prop', 'continuity', 'aesop',
  'decide', 'tauto', 'exact?', 'apply?', 'specialize', 'subst', 'left', 'right',
  'split', 'by_contra', 'by_cases', 'exfalso', 'rename_i',
]);

const IDENT = /[A-Za-z_À-ɏͰ-Ͽ℀-⅏][A-Za-z0-9_'!?.À-ɏͰ-Ͽ℀-⅏]*/y;

function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  const n = src.length;
  const push = (type: TokType, value: string) => out.push({ type, value });

  while (i < n) {
    const c = src[i];

    // Block comment /- ... -/ (nested-aware enough for our sources).
    if (c === '/' && src[i + 1] === '-') {
      let depth = 1;
      let j = i + 2;
      while (j < n && depth > 0) {
        if (src[j] === '/' && src[j + 1] === '-') { depth++; j += 2; }
        else if (src[j] === '-' && src[j + 1] === '/') { depth--; j += 2; }
        else j++;
      }
      push('comment', src.slice(i, j));
      i = j;
      continue;
    }
    // Line comment -- ...
    if (c === '-' && src[i + 1] === '-') {
      let j = i;
      while (j < n && src[j] !== '\n') j++;
      push('comment', src.slice(i, j));
      i = j;
      continue;
    }
    // String literal.
    if (c === '"') {
      let j = i + 1;
      while (j < n && src[j] !== '"') {
        if (src[j] === '\\') j++;
        j++;
      }
      j++;
      push('string', src.slice(i, Math.min(j, n)));
      i = Math.min(j, n);
      continue;
    }
    // Attribute @[...] .
    if (c === '@' && src[i + 1] === '[') {
      let j = i + 2;
      while (j < n && src[j] !== ']') j++;
      j++;
      push('attr', src.slice(i, Math.min(j, n)));
      i = Math.min(j, n);
      continue;
    }
    // Number.
    if (/[0-9]/.test(c)) {
      let j = i;
      while (j < n && /[0-9.eExXa-fA-F_]/.test(src[j])) j++;
      push('number', src.slice(i, j));
      i = j;
      continue;
    }
    // Identifier / keyword.
    IDENT.lastIndex = i;
    const m = IDENT.exec(src);
    if (m && m.index === i) {
      const word = m[0];
      if (word === 'sorry') push('sorry', word);
      else if (KEYWORDS.has(word)) push('keyword', word);
      else if (TACTICS.has(word)) push('tactic', word);
      else push('text', word);
      i += word.length;
      continue;
    }
    // Operators / unicode math symbols.
    if (/[:=→←↦∀∃∈∉⊢↑↓∘⟨⟩·≤≥≠∫⋆∑∏∧∨¬⊆⊇∩∪×|.,;(){}\[\]]/.test(c)) {
      push('op', c);
      i++;
      continue;
    }
    // Whitespace / anything else.
    let j = i;
    while (j < n && /\s/.test(src[j]) && src[j] !== '\n') j++;
    if (j > i) {
      push('text', src.slice(i, j));
      i = j;
      continue;
    }
    if (c === '\n') {
      push('text', '\n');
      i++;
      continue;
    }
    push('text', c);
    i++;
  }
  return out;
}

/** Split tokens (which may contain newlines) into per-line arrays. */
function splitLines(toks: Tok[]): Tok[][] {
  const lines: Tok[][] = [[]];
  for (const t of toks) {
    const parts = t.value.split('\n');
    for (let k = 0; k < parts.length; k++) {
      if (k > 0) lines.push([]);
      if (parts[k]) lines[lines.length - 1].push({ type: t.type, value: parts[k] });
    }
  }
  return lines;
}

export function LeanCode({
  source,
  highlight,
  onLineClick,
}: {
  source: string;
  highlight?: { start: number; end: number } | null;
  onLineClick?: (line: number) => void;
}) {
  const lines = useMemo(() => splitLines(tokenize(source)), [source]);
  return (
    <pre className="lean-code">
      <code>
        {lines.map((toks, idx) => {
          const lineNo = idx + 1;
          const hot =
            highlight && lineNo >= highlight.start && lineNo <= highlight.end;
          return (
            <div
              key={lineNo}
              className={`line${hot ? ' line-hot' : ''}`}
              onClick={onLineClick ? () => onLineClick(lineNo) : undefined}
            >
              <span className="ln">{lineNo}</span>
              <span className="lc">
                {toks.map((t, k) => (
                  <span key={k} className={`tok-${t.type}`}>
                    {t.value}
                  </span>
                ))}
              </span>
            </div>
          );
        })}
      </code>
    </pre>
  );
}
