import { useState } from 'react';
import type { Declaration } from '../types';

const KIND_COLORS: Record<string, string> = {
  theorem: '#7c9cff',
  lemma: '#7c9cff',
  def: '#4ade80',
  abbrev: '#4ade80',
  structure: '#fbbf24',
  inductive: '#fbbf24',
  class: '#fbbf24',
  instance: '#f472b6',
  example: '#a78bfa',
};

/**
 * Proof-structure outline: every declaration as a card, with its kind, a sorry
 * flag, and an expandable best-effort tactic sequence. Tapping "view" scrolls
 * the Source tab to the declaration; "run" seeds the Run tab.
 */
export function StructureOutline({
  declarations,
  onView,
  onRun,
}: {
  declarations: Declaration[];
  onView: (d: Declaration) => void;
  onRun: (d: Declaration) => void;
}) {
  if (declarations.length === 0) {
    return <p className="empty">No top-level declarations in this file.</p>;
  }
  return (
    <div className="outline">
      {declarations.map((d, i) => (
        <DeclCard key={`${d.name}-${i}`} d={d} onView={onView} onRun={onRun} />
      ))}
    </div>
  );
}

function DeclCard({
  d,
  onView,
  onRun,
}: {
  d: Declaration;
  onView: (d: Declaration) => void;
  onRun: (d: Declaration) => void;
}) {
  const [open, setOpen] = useState(false);
  const color = KIND_COLORS[d.kind] ?? '#94a3b8';
  return (
    <div className="decl-card">
      <div className="decl-head">
        <span className="kind-chip" style={{ background: color }}>{d.kind}</span>
        <span className="decl-name">{d.name}</span>
        {d.hasSorry && <span className="badge badge-sorry">sorry</span>}
      </div>
      <div className="decl-sub">
        lines {d.startLine}–{d.endLine}
        {d.tactics.length > 0 && (
          <button className="link" onClick={() => setOpen((o) => !o)}>
            {open ? 'hide' : `${d.tactics.length} tactics`}
          </button>
        )}
      </div>
      {open && d.tactics.length > 0 && (
        <ol className="tactic-list">
          {d.tactics.map((t, k) => (
            <li key={k}><code>{t}</code></li>
          ))}
        </ol>
      )}
      <div className="decl-actions">
        <button className="btn-sm" onClick={() => onView(d)}>view source</button>
        <button className="btn-sm btn-run" onClick={() => onRun(d)}>run ▶</button>
      </div>
    </div>
  );
}
