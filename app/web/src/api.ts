// Thin API client. Same-origin in production (the Node server serves this
// build); proxied to :8088 in dev (see vite.config.ts).

import type { FileDetail, FileSummary, Health, RunResult } from './types';

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  health: () => getJSON<Health>('/api/health'),

  files: () => getJSON<{ projectDir: string; files: FileSummary[] }>('/api/files'),

  file: (path: string) =>
    getJSON<FileDetail>(`/api/file?path=${encodeURIComponent(path)}`),

  async run(code: string): Promise<RunResult> {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body?.error ?? `${res.status} ${res.statusText}`);
    return body as RunResult;
  },
};

/**
 * Streamed run over WebSocket: yields status frames before the final result.
 * Falls back to the POST endpoint if the socket cannot be opened.
 */
export function runStreaming(
  code: string,
  onStatus: (stage: string) => void,
): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    let settled = false;
    let ws: WebSocket;
    try {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${proto}://${location.host}/ws`);
    } catch {
      api.run(code).then(resolve, reject);
      return;
    }
    const fail = (e: unknown) => {
      if (settled) return;
      settled = true;
      // Network/socket failure: fall back to plain POST.
      api.run(code).then(resolve, reject);
      void e;
    };
    ws.onopen = () => ws.send(JSON.stringify({ type: 'run', code }));
    ws.onerror = fail;
    ws.onclose = () => {
      if (!settled) fail(new Error('socket closed'));
    };
    ws.onmessage = (ev) => {
      let msg: any;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === 'status') onStatus(msg.stage ?? 'running');
      else if (msg.type === 'result') {
        settled = true;
        ws.close();
        resolve(msg as RunResult);
      } else if (msg.type === 'error') {
        settled = true;
        ws.close();
        reject(new Error(msg.error ?? 'run failed'));
      }
    };
  });
}
