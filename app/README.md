# Lean Proof Runner — mobile app

A mobile-first PWA for browsing, **visualising**, and **running** the Lean 4 /
Mathlib proofs in this repo (`lean/PerturbationKernel/`) from your phone.

- **Visualise** (works with no Lean toolchain): file browser, Lean
  syntax-highlighted source, and a proof-structure outline — declarations by
  kind, `sorry` flags, and a best-effort tactic sequence per proof.
- **Run** (needs a Lean backend): a scratchpad evaluated against a warm Mathlib
  environment with this library pre-imported. Typecheck `#check` / `#print`
  queries, or paste a proof containing `sorry` to see its **live goal state**.

```
app/
  server/   Node API — drives a Lean REPL + serves files & proof structure
  web/      Vite + React + TS PWA (installable, offline-capable)
```

Because Mathlib cannot compile on a phone, the architecture is a thin mobile
client talking to a backend that owns the Lean toolchain. The server can run
anywhere you keep Lean — a laptop, a workstation, or a small cloud box — and
the PWA connects to it over the network.

## Quick start (visualise-only)

No Lean needed; great for reading proofs on the couch.

```bash
# 1. build the PWA
cd app/web && npm install && npm run build

# 2. start the server (serves the build + the file API on :8088)
cd ../server && npm install && npm start
```

Open `http://<your-machine-ip>:8088` on your phone (same Wi-Fi), then use your
browser's **Add to Home Screen** to install it as an app.

## Enabling live runs

The server drives [`leanprover-community/repl`](https://github.com/leanprover-community/repl),
which keeps a warm Lean environment so Mathlib is imported only once.

```bash
# build the REPL once, against the same toolchain as this project
git clone https://github.com/leanprover-community/repl
cd repl && lake build

# point the server at the built binary
cd /path/to/perturbation-kernel/app/server
cp .env.example .env
# set REPL_BIN to e.g. /path/to/repl/.lake/build/bin/repl
npm start
```

On boot the server runs `lake env "$REPL_BIN"` inside
`lean/PerturbationKernel`, imports the library (and thus Mathlib), and reuses
that environment for every run. The first build of Mathlib is slow; subsequent
runs are fast. When `REPL_BIN` is unset the app stays in visualise-only mode.

## Dev mode

```bash
cd app/server && npm run dev      # API + WS on :8088
cd app/web   && npm run dev       # Vite on :5173, proxies /api and /ws to :8088
```

## Deployment

The PWA is static (`app/web/dist`) and can go on any static host; point its
`/api` + `/ws` at a reachable Lean backend. The simplest single-origin option
is to run `app/server` (which serves the build) on a machine that has Lean,
and expose it over HTTPS (e.g. behind a reverse proxy or a tunnel). HTTPS is
required for the PWA service worker and for `wss://` runs.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | backend + REPL mode |
| GET | `/api/files` | every Lean file + parsed proof outline |
| GET | `/api/file?path=REL` | one file's source + declarations |
| POST | `/api/run` `{code}` | typecheck a snippet → messages + goals |
| WS | `/ws` | same as `/api/run`, with streamed status frames |
