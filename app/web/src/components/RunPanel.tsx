import type { RunResult } from '../types';

/**
 * The Run tab: an editable Mathlib scratchpad evaluated against the warm
 * environment (the repo's library is already imported, so every declaration is
 * in scope). Errors and warnings come back with positions; any `sorry` in the
 * submitted code yields its open goal state — that's the "visualise the proof"
 * payoff.
 */
export function RunPanel({
  code,
  setCode,
  onRun,
  running,
  status,
  result,
  error,
  mode,
}: {
  code: string;
  setCode: (s: string) => void;
  onRun: () => void;
  running: boolean;
  status: string;
  result: RunResult | null;
  error: string | null;
  mode: 'run' | 'visualise-only';
}) {
  const disabled = running || mode !== 'run';
  return (
    <div className="run-panel">
      {mode !== 'run' && (
        <div className="notice">
          Visualise-only mode — no Lean backend is configured. Set{' '}
          <code>REPL_BIN</code> on the server to enable live runs.
        </div>
      )}
      <p className="hint">
        Scratchpad over Mathlib + this library. Try <code>#check</code>,{' '}
        <code>#print</code>, or paste a proof with <code>sorry</code> to see its
        goal state.
      </p>
      <textarea
        className="code-input"
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        value={code}
        onChange={(e) => setCode(e.target.value)}
        rows={8}
      />
      <button className="btn-primary" disabled={disabled || !code.trim()} onClick={onRun}>
        {running ? 'Running…' : 'Run ▶'}
      </button>

      {running && <div className="status">{status || 'working…'}</div>}
      {error && <div className="result-error">⚠ {error}</div>}

      {result && (
        <div className="results">
          <div className={`verdict ${result.ok ? 'ok' : 'bad'}`}>
            {result.ok ? '✓ typechecks' : '✗ errors'}
            {result.sorryCount > 0 && ` · ${result.sorryCount} open goal${result.sorryCount === 1 ? '' : 's'}`}
          </div>

          {result.goals.length > 0 && (
            <div className="goals">
              <h4>Open goals</h4>
              {result.goals.map((g, i) => (
                <div className="goal" key={i}>
                  {g.line != null && <div className="goal-pos">at line {g.line}</div>}
                  <pre>{g.goal}</pre>
                </div>
              ))}
            </div>
          )}

          {result.messages.length > 0 && (
            <div className="messages">
              <h4>Messages</h4>
              {result.messages.map((m, i) => (
                <div className={`message msg-${m.severity}`} key={i}>
                  <span className="msg-sev">{m.severity}</span>
                  {m.line != null && <span className="msg-pos">{m.line}:{m.column}</span>}
                  <pre>{m.text}</pre>
                </div>
              ))}
            </div>
          )}

          {result.messages.length === 0 && result.goals.length === 0 && (
            <p className="empty">No messages — clean.</p>
          )}
        </div>
      )}
    </div>
  );
}
