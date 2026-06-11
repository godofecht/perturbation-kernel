import { useEffect, useRef, useState } from 'react';
import { api, runStreaming } from './api';
import { LeanCode } from './lean-highlight';
import { FileList } from './components/FileList';
import { StructureOutline } from './components/StructureOutline';
import { RunPanel } from './components/RunPanel';
import type { Declaration, FileDetail, FileSummary, Health, RunResult } from './types';

type Tab = 'source' | 'structure' | 'run';

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [files, setFiles] = useState<FileSummary[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [openPath, setOpenPath] = useState<string | null>(null);
  const [detail, setDetail] = useState<FileDetail | null>(null);
  const [tab, setTab] = useState<Tab>('source');
  const [highlight, setHighlight] = useState<{ start: number; end: number } | null>(null);
  const sourceRef = useRef<HTMLDivElement>(null);

  // Run-tab state.
  const [code, setCode] = useState('#check @PerturbationKernel.PerturbationKernel');
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState('');
  const [result, setResult] = useState<RunResult | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api
      .files()
      .then((r) => setFiles(r.files))
      .catch((e) => setLoadError(String(e.message ?? e)));
  }, []);

  async function openFile(path: string) {
    setOpenPath(path);
    setDetail(null);
    setTab('source');
    setHighlight(null);
    try {
      setDetail(await api.file(path));
    } catch (e) {
      setLoadError(String((e as Error).message));
    }
  }

  function sliceDecl(d: Declaration): string {
    if (!detail) return '';
    return detail.content.split('\n').slice(d.startLine - 1, d.endLine).join('\n');
  }

  function viewDecl(d: Declaration) {
    setTab('source');
    setHighlight({ start: d.startLine, end: d.endLine });
    // Defer scroll until the source tab has rendered.
    requestAnimationFrame(() => {
      const el = sourceRef.current?.querySelector(`.line:nth-child(${d.startLine})`);
      el?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    });
  }

  function runDecl(d: Declaration) {
    const src = sliceDecl(d);
    setCode(
      `-- ${d.kind} ${d.name} (from ${openPath}). It is already imported, so\n` +
        `-- re-running verbatim may clash; edit freely, or replace its proof\n` +
        `-- with \`sorry\` to inspect the goal state.\n\n${src}\n`,
    );
    setResult(null);
    setRunError(null);
    setTab('run');
  }

  async function doRun() {
    setRunning(true);
    setRunError(null);
    setResult(null);
    setStatus('connecting…');
    try {
      const r = await runStreaming(code, setStatus);
      setResult(r);
    } catch (e) {
      setRunError(String((e as Error).message));
    } finally {
      setRunning(false);
      setStatus('');
    }
  }

  const mode = health?.mode ?? 'visualise-only';

  return (
    <div className="app">
      <header className="topbar">
        {openPath ? (
          <button className="back" onClick={() => setOpenPath(null)} aria-label="Back">
            ‹
          </button>
        ) : (
          <span className="logo" aria-hidden>⊢</span>
        )}
        <div className="title">
          {openPath ?? 'Lean Proof Runner'}
        </div>
        <span className={`mode-badge ${mode === 'run' ? 'on' : 'off'}`}>
          {mode === 'run' ? 'live' : 'view'}
        </span>
      </header>

      {!openPath && (
        <main className="home">
          {loadError && <div className="result-error">⚠ {loadError}</div>}
          {health && (
            <p className="project-line">
              Project: <code>{shortDir(health.projectDir)}</code>
            </p>
          )}
          {!files && !loadError && <p className="empty">Loading files…</p>}
          {files && <FileList files={files} onOpen={openFile} />}
        </main>
      )}

      {openPath && (
        <>
          <nav className="tabs">
            {(['source', 'structure', 'run'] as Tab[]).map((t) => (
              <button
                key={t}
                className={`tab ${tab === t ? 'active' : ''}`}
                onClick={() => setTab(t)}
              >
                {t}
              </button>
            ))}
          </nav>
          <main className="detail">
            {!detail && <p className="empty">Loading…</p>}
            {detail && tab === 'source' && (
              <div ref={sourceRef} className="source-wrap">
                <LeanCode source={detail.content} highlight={highlight} />
              </div>
            )}
            {detail && tab === 'structure' && (
              <StructureOutline
                declarations={detail.declarations}
                onView={viewDecl}
                onRun={runDecl}
              />
            )}
            {tab === 'run' && (
              <RunPanel
                code={code}
                setCode={setCode}
                onRun={doRun}
                running={running}
                status={status}
                result={result}
                error={runError}
                mode={mode}
              />
            )}
          </main>
        </>
      )}
    </div>
  );
}

function shortDir(dir: string): string {
  const parts = dir.split('/');
  return parts.slice(-2).join('/');
}
