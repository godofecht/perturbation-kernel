// HTTP + WebSocket API for the Lean proof runner PWA.
//
//   GET  /api/health           -> backend + REPL availability
//   GET  /api/files            -> manifest of every Lean file + proof outline
//   GET  /api/file?path=REL    -> one file's source + parsed declarations
//   POST /api/run  {code}      -> typecheck a snippet, return messages + goals
//   WS   /ws                   -> same as /api/run, streamed (status updates)
//
// Live runs require a built Lean REPL (REPL_BIN). Without it the file/manifest
// endpoints still work, so the PWA degrades to visualise-only.

import { readFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import http from 'node:http';
import express from 'express';
import cors from 'cors';
import { WebSocketServer } from 'ws';
import { buildManifest, readLeanFile } from './leanFiles.js';
import { LeanRepl } from './leanRepl.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// --- Config (env, with repo-relative defaults) ------------------------------
loadDotEnv(path.resolve(__dirname, '..', '.env'));

const PORT = Number(process.env.PORT ?? 8088);
const PROJECT_DIR = path.resolve(
  __dirname,
  '..',
  process.env.LEAN_PROJECT_DIR ?? '../../lean/PerturbationKernel',
);
const REPL_BIN = process.env.REPL_BIN
  ? path.resolve(__dirname, '..', process.env.REPL_BIN)
  : undefined;
const RUN_TIMEOUT_MS = Number(process.env.RUN_TIMEOUT_SECONDS ?? 120) * 1000;

const repl = new LeanRepl({
  projectDir: PROJECT_DIR,
  replBin: REPL_BIN,
  timeoutMs: RUN_TIMEOUT_MS,
});

// --- App --------------------------------------------------------------------
const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    projectDir: PROJECT_DIR,
    replConfigured: repl.available(),
    mode: repl.available() ? 'run' : 'visualise-only',
  });
});

app.get('/api/files', async (_req, res) => {
  try {
    res.json({ projectDir: PROJECT_DIR, files: await buildManifest(PROJECT_DIR) });
  } catch (e) {
    res.status(500).json({ error: String(e.message ?? e) });
  }
});

app.get('/api/file', async (req, res) => {
  const rel = String(req.query.path ?? '');
  if (!rel) return res.status(400).json({ error: 'missing ?path=' });
  try {
    res.json(await readLeanFile(PROJECT_DIR, rel));
  } catch (e) {
    res.status(404).json({ error: String(e.message ?? e) });
  }
});

app.post('/api/run', async (req, res) => {
  const code = req.body?.code;
  if (typeof code !== 'string' || !code.trim()) {
    return res.status(400).json({ error: 'body must be { code: string }' });
  }
  if (!repl.available()) {
    return res.status(503).json({
      error: 'Lean backend not configured (set REPL_BIN). Running in visualise-only mode.',
      mode: 'visualise-only',
    });
  }
  try {
    res.json(await repl.run(code));
  } catch (e) {
    res.status(500).json({ error: String(e.message ?? e) });
  }
});

// Serve the built PWA if present (single-origin deploy convenience).
const webDist = path.resolve(__dirname, '..', '..', 'web', 'dist');
app.use(express.static(webDist));
app.get('*', async (req, res, next) => {
  if (req.path.startsWith('/api') || req.path.startsWith('/ws')) return next();
  try {
    res.type('html').send(await readFile(path.join(webDist, 'index.html'), 'utf8'));
  } catch {
    res.status(404).send('Web build not found. Run `npm run build` in app/web.');
  }
});

// --- WebSocket: streamed runs with status frames ----------------------------
const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });
wss.on('connection', (ws) => {
  ws.send(JSON.stringify({ type: 'ready', mode: repl.available() ? 'run' : 'visualise-only' }));
  ws.on('message', async (data) => {
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return ws.send(JSON.stringify({ type: 'error', error: 'invalid JSON' }));
    }
    if (msg.type !== 'run' || typeof msg.code !== 'string') {
      return ws.send(JSON.stringify({ type: 'error', error: 'expected { type:"run", code }' }));
    }
    if (!repl.available()) {
      return ws.send(JSON.stringify({ type: 'error', error: 'Lean backend not configured' }));
    }
    ws.send(JSON.stringify({ type: 'status', stage: repl.proc ? 'running' : 'starting Lean + Mathlib' }));
    try {
      const result = await repl.run(msg.code);
      ws.send(JSON.stringify({ type: 'result', ...result }));
    } catch (e) {
      ws.send(JSON.stringify({ type: 'error', error: String(e.message ?? e) }));
    }
  });
});

server.listen(PORT, () => {
  console.log(`lean-proof-runner API on http://localhost:${PORT}`);
  console.log(`  project: ${PROJECT_DIR}`);
  console.log(`  mode:    ${repl.available() ? 'run (REPL configured)' : 'visualise-only (set REPL_BIN to enable runs)'}`);
});

process.on('SIGINT', () => {
  repl.shutdown();
  process.exit(0);
});

// Minimal .env loader (no dependency): KEY=VALUE lines, # comments. Real
// environment variables always win over the file.
function loadDotEnv(file) {
  let text;
  try {
    text = readFileSync(file, 'utf8');
  } catch {
    return; // no .env present — rely on the real environment
  }
  for (const line of text.split('\n')) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/);
    if (m && process.env[m[1]] === undefined) {
      process.env[m[1]] = m[2].replace(/^['"]|['"]$/g, '');
    }
  }
}
