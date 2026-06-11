"use client";

import { useMemo } from "react";
import { COOKS, MEALS, NEIGHBORHOODS } from "@/lib/data";
import { matchMeals } from "@/lib/matching";
import { useStore } from "@/lib/store";
import type { Cuisine, Dietary, Sitting } from "@/lib/types";
import { MealCard } from "@/components/MealCard";

const CUISINES: Cuisine[] = [
  "italian",
  "indian",
  "japanese",
  "mexican",
  "lebanese",
  "thai",
  "ethiopian",
  "french",
  "korean",
  "vegan-soul",
];

const DIETARY: Dietary[] = [
  "vegetarian",
  "vegan",
  "halal",
  "gluten-free",
  "nut-free",
  "dairy-free",
];

const SITTINGS: (Sitting | "any")[] = ["any", "lunch", "early-dinner", "late-dinner"];

function toggle<T>(list: T[], item: T): T[] {
  return list.includes(item) ? list.filter((x) => x !== item) : [...list, item];
}

export default function MatchPage() {
  const { prefs, setPrefs } = useStore();

  const ranked = useMemo(
    () => matchMeals(MEALS, COOKS, prefs),
    [prefs],
  );

  return (
    <>
      <section className="section">
        <h2>Find my table</h2>
        <p className="sub">
          Tell us what you fancy. We rank tonight&apos;s cooks on cuisine, diet,
          distance, reputation and budget — and flag tables with seats to share.
        </p>
      </section>

      <div className="detail-grid">
        <div className="panel">
          <div className="field">
            <label>
              What are you in the mood for?{" "}
              <span className="hint">pick any cuisines you like</span>
            </label>
            <div className="chips">
              {CUISINES.map((c) => (
                <span
                  key={c}
                  className={`chip ${prefs.cuisines.includes(c) ? "on" : ""}`}
                  onClick={() =>
                    setPrefs({ ...prefs, cuisines: toggle(prefs.cuisines, c) })
                  }
                >
                  {c.replace("-", " ")}
                </span>
              ))}
            </div>
          </div>

          <div className="field">
            <label>
              Dietary needs{" "}
              <span className="hint">meals must satisfy all of these</span>
            </label>
            <div className="chips">
              {DIETARY.map((d) => (
                <span
                  key={d}
                  className={`chip diet ${prefs.dietary.includes(d) ? "on" : ""}`}
                  onClick={() =>
                    setPrefs({ ...prefs, dietary: toggle(prefs.dietary, d) })
                  }
                >
                  {d}
                </span>
              ))}
            </div>
          </div>

          <div className="row">
            <div className="field">
              <label>Your neighborhood</label>
              <select
                value={prefs.neighborhood}
                onChange={(e) =>
                  setPrefs({ ...prefs, neighborhood: e.target.value })
                }
              >
                {NEIGHBORHOODS.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>When</label>
              <select
                value={prefs.sitting}
                onChange={(e) =>
                  setPrefs({ ...prefs, sitting: e.target.value as Sitting | "any" })
                }
              >
                {SITTINGS.map((s) => (
                  <option key={s} value={s}>
                    {s === "any" ? "Any time" : s.replace("-", " ")}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="field">
            <label>
              Budget per seat: <strong>£{prefs.budget}</strong>
            </label>
            <input
              type="range"
              min={8}
              max={30}
              value={prefs.budget}
              onChange={(e) =>
                setPrefs({ ...prefs, budget: Number(e.target.value) })
              }
            />
          </div>

          <div className="field" style={{ marginBottom: 0 }}>
            <label className="spread" style={{ cursor: "pointer" }}>
              <span>
                Match me with meal buddies{" "}
                <span className="hint">prefer shared, communal tables</span>
              </span>
              <input
                type="checkbox"
                checked={prefs.wantsBuddies}
                onChange={(e) =>
                  setPrefs({ ...prefs, wantsBuddies: e.target.checked })
                }
              />
            </label>
          </div>
        </div>

        <div className="stack">
          <div className="callout">
            <strong>{ranked.length}</strong> meal{ranked.length === 1 ? "" : "s"}{" "}
            match your needs
            {prefs.dietary.length > 0 && (
              <> · {prefs.dietary.join(", ")} guaranteed</>
            )}
            .
          </div>
          {ranked.length === 0 && (
            <div className="empty">
              No meals clear your dietary filter tonight. Try relaxing a
              requirement.
            </div>
          )}
        </div>
      </div>

      <section className="section">
        <h2>Your ranked tables</h2>
        <p className="sub">Best match first. Tap a meal to see the cook and buddies.</p>
        <div className="grid">
          {ranked.map((m) => (
            <MealCard key={m.meal.id} meal={m.meal} score={m.score} />
          ))}
        </div>
      </section>
    </>
  );
}
