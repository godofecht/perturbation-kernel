// Shared shapes mirroring the server's JSON contract (app/server/src).

export interface Declaration {
  kind: string;
  name: string;
  startLine: number;
  endLine: number;
  hasSorry: boolean;
  tactics: string[];
}

export interface FileSummary {
  path: string;
  lines: number;
  declarations: Declaration[];
  sorryCount: number;
}

export interface FileDetail {
  path: string;
  content: string;
  declarations: Declaration[];
}

export interface RunMessage {
  severity: 'error' | 'warning' | 'info' | string;
  line: number | null;
  column: number | null;
  endLine: number | null;
  text: string;
}

export interface RunGoal {
  line: number | null;
  column: number | null;
  goal: string;
}

export interface RunResult {
  ok: boolean;
  messages: RunMessage[];
  goals: RunGoal[];
  sorryCount: number;
  env: number | null;
}

export interface Health {
  ok: boolean;
  projectDir: string;
  replConfigured: boolean;
  mode: 'run' | 'visualise-only';
}
