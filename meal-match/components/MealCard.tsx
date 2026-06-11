import Link from "next/link";
import { cookById } from "@/lib/data";
import type { Meal } from "@/lib/types";
import { ScoreBadge } from "./ScoreBadge";

export function MealCard({ meal, score }: { meal: Meal; score?: number }) {
  const cook = cookById(meal.cookId);
  const seatsLeft = meal.seatsTotal - meal.seatsTaken.length;
  return (
    <Link href={`/meals/${meal.id}`} className="card">
      <div className="thumb">{meal.image}</div>
      <div className="body">
        <div className="spread">
          <span className="title">{meal.title}</span>
          {typeof score === "number" && <ScoreBadge score={score} />}
        </div>
        <div className="desc">{meal.description}</div>
        <div className="chips">
          <span className="tag">{meal.cuisine.replace("-", " ")}</span>
          {meal.dietary.slice(0, 2).map((d) => (
            <span className="tag" key={d}>
              {d}
            </span>
          ))}
        </div>
        <div className="foot">
          <span className="who">
            <span style={{ fontSize: 20 }}>{cook?.avatar}</span>
            <span className="meta">
              <span className="small">{cook?.name}</span>
            </span>
          </span>
          <span className="price">£{meal.price}</span>
        </div>
        <div className="foot" style={{ paddingTop: 0 }}>
          <span className="tag">📍 {meal.neighborhood}</span>
          {meal.communal ? (
            <span className="tag">
              {seatsLeft > 0 ? `${seatsLeft} seats left` : "table full"}
            </span>
          ) : (
            <span className="tag">solo plate</span>
          )}
        </div>
      </div>
    </Link>
  );
}
