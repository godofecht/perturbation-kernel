import { describe, expect, it } from "vitest";
import { COOKS, DINERS, MEALS, cookById, mealById } from "./data";
import {
  matchBuddies,
  matchMeals,
  meetsDietary,
  proximity,
  scoreMeal,
} from "./matching";
import type { Preferences } from "./types";

const basePrefs: Preferences = {
  cuisines: ["indian", "ethiopian"],
  dietary: ["vegan"],
  neighborhood: "Brixton",
  sitting: "any",
  budget: 20,
  wantsBuddies: true,
};

describe("proximity", () => {
  it("scores same neighborhood highest", () => {
    expect(proximity("Peckham", "Peckham")).toBe(1);
  });
  it("scores adjacent neighborhoods in between", () => {
    expect(proximity("Peckham", "Deptford")).toBe(0.5);
  });
  it("scores distant neighborhoods zero", () => {
    expect(proximity("Peckham", "Shoreditch")).toBe(0);
  });
});

describe("meetsDietary", () => {
  it("passes when the meal covers every requirement", () => {
    const sadya = mealById("meal-avial")!;
    expect(meetsDietary(sadya, ["vegan", "gluten-free"])).toBe(true);
  });
  it("fails when a requirement is missing", () => {
    const tortelli = mealById("meal-tortelli")!; // vegetarian only
    expect(meetsDietary(tortelli, ["vegan"])).toBe(false);
  });
});

describe("scoreMeal", () => {
  it("disqualifies meals that clash with dietary needs", () => {
    const tortelli = mealById("meal-tortelli")!;
    const cook = cookById(tortelli.cookId)!;
    const result = scoreMeal(tortelli, cook, basePrefs);
    expect(result.disqualified).toBe(true);
    expect(result.score).toBe(0);
  });

  it("rewards a meal that hits cuisine, diet, and location", () => {
    const sadya = mealById("meal-avial")!; // indian, vegan, Brixton
    const cook = cookById(sadya.cookId)!;
    const result = scoreMeal(sadya, cook, basePrefs);
    expect(result.disqualified).toBe(false);
    expect(result.score).toBeGreaterThan(80);
    expect(result.reasons.some((r) => r.label.includes("Brixton"))).toBe(true);
  });

  it("keeps scores within 0..100", () => {
    for (const meal of MEALS) {
      const cook = cookById(meal.cookId)!;
      const { score } = scoreMeal(meal, cook, basePrefs);
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(100);
    }
  });
});

describe("matchMeals", () => {
  it("drops disqualified meals and sorts best-first", () => {
    const ranked = matchMeals(MEALS, COOKS, basePrefs);
    expect(ranked.every((m) => !m.disqualified)).toBe(true);
    for (let i = 1; i < ranked.length; i++) {
      expect(ranked[i - 1].score).toBeGreaterThanOrEqual(ranked[i].score);
    }
  });

  it("puts the Keralan vegan sadya at the top for this diner", () => {
    const ranked = matchMeals(MEALS, COOKS, basePrefs);
    expect(ranked[0].meal.id).toBe("meal-avial");
  });
});

describe("matchBuddies", () => {
  it("excludes diners already seated at the table", () => {
    const sadya = mealById("meal-avial")!; // jade, fatima, nina seated
    const buddies = matchBuddies(sadya, DINERS);
    const ids = buddies.map((b) => b.diner.id);
    expect(ids).not.toContain("diner-jade");
    expect(ids).not.toContain("diner-fatima");
  });

  it("ranks a same-cuisine, nearby diner above an unrelated one", () => {
    const mole = mealById("meal-mole")!; // mexican, Deptford
    const buddies = matchBuddies(mole, DINERS);
    expect(buddies[0].score).toBeGreaterThanOrEqual(buddies[buddies.length - 1].score);
    // Kwame likes mexican and lives in Deptford -> strong buddy.
    expect(buddies[0].diner.id).toBe("diner-kwame");
  });

  it("honours additional exclusions", () => {
    const mole = mealById("meal-mole")!;
    const buddies = matchBuddies(mole, DINERS, ["diner-kwame"]);
    expect(buddies.map((b) => b.diner.id)).not.toContain("diner-kwame");
  });
});
