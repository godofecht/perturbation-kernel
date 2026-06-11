import { ADJACENCY } from "./data";
import type { Cook, Diner, Meal, Preferences } from "./types";

// The matching engine. Two related jobs:
//   1. matchMeals  — rank tonight's meals (and their cooks) for a diner.
//   2. matchBuddies — for a chosen meal, rank fellow diners to share it with.
//
// Everything is a transparent, weighted sum so the UI can show *why* a match
// scored well. Scores are normalised to roughly 0..100.

export interface ScoredReason {
  label: string;
  /** Positive = boosted the match, negative = dragged it down. */
  delta: number;
}

export interface MealMatch {
  meal: Meal;
  cook: Cook;
  score: number; // 0..100, higher is better
  reasons: ScoredReason[];
  /** A hard incompatibility (e.g. dietary clash) that excludes the meal. */
  disqualified: boolean;
}

export interface BuddyMatch {
  diner: Diner;
  score: number; // 0..100
  sharedCuisines: string[];
}

const WEIGHTS = {
  cuisine: 34, // overlap between diner tastes and the meal's cuisine
  dietary: 26, // meal must satisfy declared dietary needs
  location: 18, // same / adjacent neighborhood
  rating: 12, // cook reputation
  budget: 10, // comfortably within budget
};

function jaccard<T>(a: T[], b: T[]): number {
  if (a.length === 0 && b.length === 0) return 0;
  const setB = new Set(b);
  const inter = a.filter((x) => setB.has(x)).length;
  const union = new Set([...a, ...b]).size;
  return union === 0 ? 0 : inter / union;
}

/** 1 if same neighborhood, 0.5 if adjacent, 0 otherwise. */
export function proximity(a: string, b: string): number {
  if (a === b) return 1;
  if ((ADJACENCY[a] ?? []).includes(b)) return 0.5;
  return 0;
}

/**
 * Does the meal satisfy every dietary requirement the diner declared?
 * Missing a required guarantee is a hard fail, not a soft penalty — we won't
 * serve a vegan a "vegetarian" meal and hope for the best.
 */
export function meetsDietary(meal: Meal, required: Preferences["dietary"]): boolean {
  return required.every((tag) => meal.dietary.includes(tag));
}

export function scoreMeal(meal: Meal, cook: Cook, prefs: Preferences): MealMatch {
  const reasons: ScoredReason[] = [];

  const dietaryOk = meetsDietary(meal, prefs.dietary);
  if (!dietaryOk) {
    return {
      meal,
      cook,
      score: 0,
      disqualified: true,
      reasons: [{ label: "Doesn't meet your dietary needs", delta: 0 }],
    };
  }

  let score = 0;

  // Cuisine: how well the meal's cuisine sits inside the diner's tastes.
  const cuisineHit = prefs.cuisines.includes(meal.cuisine);
  const cuisinePts = cuisineHit ? WEIGHTS.cuisine : WEIGHTS.cuisine * 0.15;
  score += cuisinePts;
  reasons.push({
    label: cuisineHit
      ? `Matches your taste for ${meal.cuisine.replace("-", " ")}`
      : "Outside your usual cuisines",
    delta: Math.round(cuisinePts),
  });

  // Dietary: passing the hard gate earns the full weight; bonus tags noted.
  if (prefs.dietary.length > 0) {
    score += WEIGHTS.dietary;
    reasons.push({ label: "Meets every dietary requirement", delta: WEIGHTS.dietary });
  } else {
    score += WEIGHTS.dietary * 0.5;
  }

  // Location: same or adjacent neighborhood.
  const prox = proximity(prefs.neighborhood, meal.neighborhood);
  const locPts = WEIGHTS.location * prox;
  score += locPts;
  if (prox === 1) {
    reasons.push({ label: `In ${meal.neighborhood} — your area`, delta: Math.round(locPts) });
  } else if (prox === 0.5) {
    reasons.push({ label: `Next door in ${meal.neighborhood}`, delta: Math.round(locPts) });
  } else {
    reasons.push({ label: `Across town in ${meal.neighborhood}`, delta: 0 });
  }

  // Rating: scaled cook reputation.
  const ratingPts = WEIGHTS.rating * (cook.rating / 5);
  score += ratingPts;
  reasons.push({ label: `Cook rated ${cook.rating.toFixed(1)}★`, delta: Math.round(ratingPts) });

  // Budget: full points if within budget, scaled penalty if over.
  if (meal.price <= prefs.budget) {
    score += WEIGHTS.budget;
    reasons.push({ label: `£${meal.price} — within budget`, delta: WEIGHTS.budget });
  } else {
    const over = meal.price - prefs.budget;
    const penalty = Math.min(WEIGHTS.budget, over * 2);
    score -= penalty;
    reasons.push({ label: `£${meal.price} — over budget`, delta: -Math.round(penalty) });
  }

  // Sitting / time window: soft preference, not in the weighted base.
  if (prefs.sitting !== "any") {
    if (meal.sitting === prefs.sitting) {
      score += 6;
      reasons.push({ label: `Served at ${labelSitting(meal.sitting)}`, delta: 6 });
    } else {
      score -= 4;
      reasons.push({ label: `Served at ${labelSitting(meal.sitting)}, not your slot`, delta: -4 });
    }
  }

  // Communal bonus when the diner wants buddies and the table is open & has room.
  if (prefs.wantsBuddies && meal.communal) {
    const seatsLeft = meal.seatsTotal - meal.seatsTaken.length;
    if (seatsLeft > 0) {
      score += 8;
      reasons.push({ label: `${seatsLeft} seats left to share`, delta: 8 });
    } else {
      reasons.push({ label: "Table is full", delta: 0 });
    }
  }

  return {
    meal,
    cook,
    score: clamp(Math.round(score)),
    disqualified: false,
    reasons: reasons.filter((r) => r.delta !== 0),
  };
}

/** Rank all meals for a diner, best first, with disqualified meals dropped. */
export function matchMeals(
  meals: Meal[],
  cooks: Cook[],
  prefs: Preferences,
): MealMatch[] {
  const cookMap = new Map(cooks.map((c) => [c.id, c]));
  return meals
    .map((meal) => {
      const cook = cookMap.get(meal.cookId);
      if (!cook) return null;
      return scoreMeal(meal, cook, prefs);
    })
    .filter((m): m is MealMatch => m !== null && !m.disqualified)
    .sort((a, b) => b.score - a.score);
}

/**
 * For a chosen meal, rank fellow diners as meal buddies. Buddies must be
 * compatible with the meal's own dietary guarantees, and are scored on taste
 * overlap plus living nearby. Diners already seated are skipped.
 */
export function matchBuddies(
  meal: Meal,
  diners: Diner[],
  excludeIds: string[] = [],
): BuddyMatch[] {
  const skip = new Set([...meal.seatsTaken, ...excludeIds]);
  return diners
    .filter((d) => !skip.has(d.id))
    .map((diner) => {
      const sharedCuisines = diner.tastes.filter((t) => t === meal.cuisine);
      // Taste overlap is measured against the meal's single cuisine plus a
      // broader affinity for the kinds of food this diner likes.
      const cuisineScore = diner.tastes.includes(meal.cuisine) ? 1 : 0;
      const tasteBreadth = jaccard(diner.tastes, [meal.cuisine]);
      const prox = proximity(diner.neighborhood, meal.neighborhood);

      const score = clamp(
        Math.round(60 * cuisineScore + 25 * prox + 15 * tasteBreadth),
      );
      return { diner, score, sharedCuisines };
    })
    .sort((a, b) => b.score - a.score);
}

export function labelSitting(s: string): string {
  switch (s) {
    case "lunch":
      return "lunch";
    case "early-dinner":
      return "early dinner";
    case "late-dinner":
      return "late dinner";
    default:
      return s;
  }
}

function clamp(n: number): number {
  return Math.max(0, Math.min(100, n));
}
