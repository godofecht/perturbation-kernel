"use client";

import Link from "next/link";
import { cookById, dinerById, mealById } from "@/lib/data";
import { useStore } from "@/lib/store";

export default function TablesPage() {
  const { joined, leaveTable } = useStore();

  return (
    <>
      <section className="section">
        <h2>My tables</h2>
        <p className="sub">Meals you&apos;ve joined and the buddies you invited.</p>
      </section>

      {joined.length === 0 ? (
        <div className="empty">
          You haven&apos;t joined a table yet.{" "}
          <Link href="/match" style={{ color: "var(--accent)", fontWeight: 700 }}>
            Find your table →
          </Link>
        </div>
      ) : (
        <div className="stack">
          {joined.map(({ mealId, buddyIds }) => {
            const meal = mealById(mealId);
            if (!meal) return null;
            const cook = cookById(meal.cookId);
            return (
              <div className="panel" key={mealId}>
                <div className="spread">
                  <div className="who">
                    <span className="face">{meal.image}</span>
                    <span className="meta">
                      <Link href={`/meals/${meal.id}`} className="name">
                        {meal.title}
                      </Link>
                      <div className="small">
                        {cook?.name} · 📍 {meal.neighborhood} · £{meal.price}
                      </div>
                    </span>
                  </div>
                  <button className="btn ghost" onClick={() => leaveTable(mealId)}>
                    Leave
                  </button>
                </div>

                {buddyIds.length > 0 && (
                  <div className="chips" style={{ marginTop: 14 }}>
                    <span className="small" style={{ color: "var(--muted)" }}>
                      Dining with:
                    </span>
                    {buddyIds.map((id) => {
                      const b = dinerById(id);
                      return b ? (
                        <span className="tag" key={id}>
                          {b.avatar} {b.name}
                        </span>
                      ) : null;
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}
