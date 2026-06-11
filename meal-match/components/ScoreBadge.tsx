export function ScoreBadge({ score }: { score: number }) {
  const cls = score >= 75 ? "" : score >= 50 ? "mid" : "low";
  return (
    <span className={`score ${cls}`} title="Match score">
      {score}% match
    </span>
  );
}
