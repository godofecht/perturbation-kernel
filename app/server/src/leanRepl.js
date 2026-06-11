// Persistent Lean 4 / Mathlib REPL manager.
//
// We drive https://github.com/leanprover-community/repl : a small program that
// reads JSON commands on stdin and writes JSON responses on stdout, keeping a
// warm Lean environment between commands so Mathlib is imported only once.
//
// Protocol (repl): each request and each response is a JSON object terminated
// by a blank line. A command request looks like
//     { "cmd": "<lean source>", "env": <previous env id or omitted> }
// and the response carries `messages` (errors/warnings/info with positions),
// `sorries` (each with the open `goal` and its position), and a new `env` id.
//
// We serialise requests through a queue because the REPL is single-threaded
// over one stdin/stdout pair.

import { spawn } from 'node:child_process';

export class LeanRepl {
  /**
   * @param {object} opts
   * @param {string} opts.projectDir  cwd for `lake env` (the Mathlib project).
   * @param {string|undefined} opts.replBin  path to the built repl binary.
   * @param {number} opts.timeoutMs  per-command timeout.
   */
  constructor({ projectDir, replBin, timeoutMs }) {
    this.projectDir = projectDir;
    this.replBin = replBin;
    this.timeoutMs = timeoutMs ?? 120_000;
    this.proc = null;
    this.buffer = '';
    this.queue = [];
    this.current = null;
    // env id 0 is established by importing Mathlib + the local library once.
    this.baseEnv = null;
    this.starting = null;
  }

  available() {
    return Boolean(this.replBin);
  }

  /** Lazily spawn the REPL and import Mathlib. Idempotent. */
  async ensureStarted() {
    if (!this.available()) throw new Error('REPL_BIN not configured');
    if (this.proc) return;
    if (this.starting) return this.starting;
    this.starting = this._start();
    try {
      await this.starting;
    } finally {
      this.starting = null;
    }
  }

  async _start() {
    this.proc = spawn('lake', ['env', this.replBin], {
      cwd: this.projectDir,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.proc.stdout.setEncoding('utf8');
    this.proc.stderr.setEncoding('utf8');
    this.proc.stdout.on('data', (chunk) => this._onStdout(chunk));
    this.proc.stderr.on('data', (chunk) => {
      // Surface REPL stderr for debugging; not part of the protocol.
      process.stderr.write(`[repl] ${chunk}`);
    });
    this.proc.on('exit', (code) => {
      const err = new Error(`Lean REPL exited (code ${code})`);
      if (this.current) this.current.reject(err);
      for (const q of this.queue) q.reject(err);
      this.queue = [];
      this.current = null;
      this.proc = null;
      this.baseEnv = null;
    });

    // Warm the environment: import the whole library (which pulls Mathlib).
    const res = await this._send({ cmd: 'import PerturbationKernel' });
    this.baseEnv = typeof res.env === 'number' ? res.env : 0;
  }

  /**
   * Typecheck a Lean snippet against the warm Mathlib environment. Returns the
   * normalised result consumed by the API/UI.
   */
  async run(code) {
    await this.ensureStarted();
    const res = await this._send({ cmd: code, env: this.baseEnv });
    return normaliseResult(res);
  }

  _send(request) {
    return new Promise((resolve, reject) => {
      this.queue.push({ request, resolve, reject });
      this._drain();
    });
  }

  _drain() {
    if (this.current || this.queue.length === 0 || !this.proc) return;
    this.current = this.queue.shift();
    const payload = JSON.stringify(this.current.request) + '\n\n';
    this.current.timer = setTimeout(() => {
      const c = this.current;
      this.current = null;
      if (c) c.reject(new Error(`Lean run timed out after ${this.timeoutMs}ms`));
      this._drain();
    }, this.timeoutMs);
    this.proc.stdin.write(payload);
  }

  _onStdout(chunk) {
    this.buffer += chunk;
    // Responses are separated by a blank line ("\n\n").
    let sep;
    while ((sep = this.buffer.indexOf('\n\n')) !== -1) {
      const raw = this.buffer.slice(0, sep).trim();
      this.buffer = this.buffer.slice(sep + 2);
      if (!raw) continue;
      const c = this.current;
      if (!c) continue;
      clearTimeout(c.timer);
      this.current = null;
      try {
        c.resolve(JSON.parse(raw));
      } catch (e) {
        c.reject(new Error(`Bad JSON from REPL: ${e.message}`));
      }
      this._drain();
    }
  }

  shutdown() {
    if (this.proc) {
      this.proc.kill();
      this.proc = null;
    }
  }
}

/** Shape the raw REPL response into the contract the frontend expects. */
function normaliseResult(res) {
  const messages = (res.messages ?? []).map((m) => ({
    severity: m.severity ?? 'info',
    line: m.pos?.line ?? null,
    column: m.pos?.column ?? null,
    endLine: m.endPos?.line ?? null,
    text: m.data ?? '',
  }));
  const goals = (res.sorries ?? []).map((s) => ({
    line: s.pos?.line ?? null,
    column: s.pos?.column ?? null,
    goal: s.goal ?? '',
  }));
  const hasError = messages.some((m) => m.severity === 'error');
  return {
    ok: !hasError,
    messages,
    goals,
    sorryCount: goals.length,
    env: res.env ?? null,
  };
}
