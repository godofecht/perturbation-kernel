"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { DINERS, cookById, mealById } from "@/lib/data";
import { matchBuddies, scoreMeal } from "@/lib/matching";
import { useStore } from "@/lib/store";
import { ScoreBadge } from "@/components/ScoreBadge";

export default function MealPage({ params }: { params: { id: string } }) {
  const meal = mealById(params.id);
  const { prefs, isJoined, joinTable, leaveTable } = useStore();
  const [picked, setPicked] = useState<string[]>([]);

  const cook = meal ? cookById(meal.cookId) : undefined;

  const match = useMemo(
    () => (meal && cook ? scoreMeal(meal, cook, prefs) : null),
    [meal, cook, prefs],
  );

  const buddies = useMemo(
    () => (meal ? matchBuddies(meal, DINERS) : []),
    [meal],
  );

  if (!meal || !cook) {
    return (
      <section className="section">
        <Link href="/" className="back">
          ← Back
        </Link>
        <div className="empty">That meal doesn&apos;t exist.</div>
      </section>
    );
  }

  const joined = isJoined(meal.id);
  const seatsLeft = meal.seatsTotal - meal.seatsTaken.length - (joined ? 1 : 0);

  function togglePick(id: string) {
    setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));
  }

  return (
    <>
      <Link href="/match" className="back">
        ← Back to matches
      </Link>

      <div className="detail-grid">
        {/* Left: the meal + cook */}
        <div className="stack">
          <div className="banner">{meal.image}</div>

          <div className="spread">
            <h2 style={{ margin: 0 }}>{meal.title}</h2>
            {match && !match.disqualified && <ScoreBadge score={match.score} />}
          </div>
          <p className="sub" style={{ margin: 0 }}>
            {meal.description}
          </p>
          <div className="chips">
            <span className="tag">{meal.cuisine.replace("-", " ")}</span>
            <span className="tag">📍 {meal.neighborhood}</span>
            <span className="tag">£{meal.price} / seat</span>
            {meal.dietary.map((d) => (
              <span className="tag" key={d}>
                {d}
              </span>
            ))}
          </div>

          <div className="panel">
            <div className="who">
              <span className="face">{cook.avatar}</span>
              <span className="meta">
                <span className="name">{cook.name}</span>
                <div className="small">
                  {cook.rating.toFixed(1)}★ · {cook.reviews} reviews ·{" "}
                  {cook.neighborhood}
                </div>
              </span>
            </div>
            <p className="desc" style={{ marginTop: 12 }}>
              {cook.bio}
            </p>
            <div className="small" style={{ color: "var(--muted)" }}>
              Signature: {cook.signatureDish}
            </div>
          </div>

          {match && !match.disqualified && match.reasons.length > 0 && (
            <div className="panel">
              <div style={{ fontWeight: 700, marginBottom: 10 }}>
                Why this is a {match.score}% match
              </div>
              <div className="reasons">
                {match.reasons.map((r, i) => (
                  <div className="reason" key={i}>
                    <span>{r.label}</span>
                    <span className={r.delta >= 0 ? "pos" : "neg"}>
                      {r.delta >= 0 ? `+${r.delta}` : r.delta}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: the table + buddies */}
        <div className="stack">
          <div className="panel">
            <div className="spread">
              <strong>The table</strong>
              <span className="tag">
                {meal.communal ? "communal" : "solo plate"}
              </span>
            </div>
            <div className="seats" style={{ marginTop: 12 }}>
              {Array.from({ length: meal.seatsTotal }).map((_, i) => {
                const taken = i < meal.seatsTaken.length;
                const mine = !taken && joined && i === meal.seatsTaken.length;
                return (
                  <span
                    key={i}
                    className={`seat ${taken || mine ? "full" : ""}`}
                    title={taken ? "taken" : mine ? "you" : "open"}
                  >
                    {taken ? "🍽️" : mine ? "🫵" : ""}
                  </span>
                );
              })}
            </div>
            <div className="small" style={{ color: "var(--muted)", marginTop: 10 }}>
              {seatsLeft > 0
                ? `${seatsLeft} of ${meal.seatsTotal} seats open`
                : "Table is full"}
            </div>

            {!joined ? (
              <button
                className="btn primary"
                style={{ width: "100%", marginTop: 14, justifyContent: "center" }}
                disabled={!meal.communal && meal.seatsTaken.length >= meal.seatsTotal}
                onClick={() => joinTable(meal.id, picked)}
              >
                Join this table{picked.length ? ` with ${picked.length} buddy${picked.length > 1 ? "s" : ""}` : ""}
              </button>
            ) : (
              <button
                className="btn good"
                style={{ width: "100%", marginTop: 14, justifyContent: "center" }}
                onClick={() => leaveTable(meal.id)}
              >
                ✓ You&apos;re in — tap to leave
              </button>
            )}
          </div>

          {meal.communal && (
            <div className="panel">
              <div className="spread" style={{ marginBottom: 12 }}>
                <strong>Meal buddies</strong>
                <span className="hint">ranked by shared taste &amp; distance</span>
              </div>
              {buddies.length === 0 && (
                <div className="small" style={{ color: "var(--muted)" }}>
                  Everyone nearby is already seated.
                </div>
              )}
              <div className="stack">
                {buddies.map(({ diner, score, sharedCuisines }) => {
                  const isPicked = picked.includes(diner.id);
                  return (
                    <div
                      key={diner.id}
                      className={`buddy ${isPicked ? "picked" : ""}`}
                    >
                      <span className="face">{diner.avatar}</span>
                      <span className="meta">
                        <span className="name">
                          {diner.name}{" "}
                          <span
                            className="small"
                            style={{ color: "var(--muted)", fontWeight: 400 }}
                          >
                            · {score}%
                          </span>
                        </span>
                        <div className="small">{diner.bio}</div>
                        {sharedCuisines.length > 0 && (
                          <div className="small" style={{ color: "var(--good)" }}>
                            both love {sharedCuisines.join(", ").replace("-", " ")}
                          </div>
                        )}
                      </span>
                      <button
                        className={`btn pick ${isPicked ? "good" : "ghost"}`}
                        onClick={() => togglePick(diner.id)}
                      >
                        {isPicked ? "✓ invited" : "invite"}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
