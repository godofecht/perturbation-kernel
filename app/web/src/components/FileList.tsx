import type { FileSummary } from '../types';

export function FileList({
  files,
  onOpen,
}: {
  files: FileSummary[];
  onOpen: (path: string) => void;
}) {
  return (
    <ul className="file-list">
      {files.map((f) => {
        const declCount = f.declarations.length;
        return (
          <li key={f.path}>
            <button className="file-card" onClick={() => onOpen(f.path)}>
              <div className="file-path">{f.path}</div>
              <div className="file-meta">
                <span>{declCount} decl{declCount === 1 ? '' : 's'}</span>
                <span>{f.lines} lines</span>
                {f.sorryCount > 0 && (
                  <span className="badge badge-sorry">{f.sorryCount} sorry</span>
                )}
                {f.sorryCount === 0 && declCount > 0 && (
                  <span className="badge badge-ok">complete</span>
                )}
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
