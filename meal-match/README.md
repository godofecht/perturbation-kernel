# MealMatch

A Deliveroo alternative built around **home cooks** and **shared tables**
instead of restaurants and couriers. You tell MealMatch what you fancy and it:

1. **Matches you with a cook** — ranking tonight's home-cooked meals on cuisine,
   dietary fit, distance, cook reputation, and budget.
2. **Matches you with meal buddies** — for any communal table, it ranks nearby
   diners with overlapping taste so you can fill the table with good company.

This is a self-contained clickable prototype (Next.js + TypeScript, no backend).
It lives in its own subdirectory and is unrelated to the Rust/Lean
`perturbation-kernel` library in the repository root.

## Run it

```bash
cd meal-match
npm install
npm run dev      # http://localhost:3000
```

Other scripts:

```bash
npm test         # vitest — the matching-engine unit tests
npm run build    # production build (also type-checks + lints)
npm start        # serve the production build
```

Requires Node 18+ (developed on Node 22).

## What's in the demo

| Route | What it does |
|-------|--------------|
| `/` | Landing page + browse tonight's cooks and meals |
| `/match` | Preferences form (cuisine, diet, area, time, budget, buddies) with **live ranked results** |
| `/meals/[id]` | Meal + cook detail, a visual **table of seats**, an explainable match breakdown, and **ranked meal buddies** you can invite |
| `/tables` | The tables you've joined and the buddies you brought |

Joining a table and inviting buddies is persisted to `localStorage`, so the demo
survives a refresh. Six cooks, seven meals, and six diners ship as seed data in
[`lib/data.ts`](lib/data.ts).

## The matching engine

All matching lives in [`lib/matching.ts`](lib/matching.ts) and is deliberately a
transparent, weighted sum so the UI can show *why* a meal scored well.

**Meal ranking** (`matchMeals` / `scoreMeal`):

- **Dietary fit is a hard gate** — a meal that doesn't satisfy *every* declared
  dietary requirement is disqualified, never soft-ranked. We won't offer a vegan
  a merely "vegetarian" dish.
- Cuisine overlap, neighborhood proximity (same / adjacent / across town), cook
  rating, and budget headroom each contribute weighted points.
- Soft nudges for the right sitting (lunch / early / late dinner) and for
  communal tables with seats left when you want buddies.
- Every meal carries a list of human-readable `reasons` (`+12 In Brixton — your
  area`) rendered on the detail page.

**Buddy ranking** (`matchBuddies`):

- Skips diners already seated (and anyone you exclude).
- Scores on shared cuisine taste plus living nearby.

The engine is covered by 13 unit tests in
[`lib/matching.test.ts`](lib/matching.test.ts) (`npm test`).

## Architecture notes

- **Next.js 14 App Router**, TypeScript, plain CSS (no UI framework) for a
  dependency-light, fast build.
- Client state (preferences + joined tables) is a small React context store in
  [`lib/store.tsx`](lib/store.tsx), persisted to `localStorage`. No server or
  database — swapping `lib/data.ts` for API calls is the natural next step.

### Security note

`npm audit` reports advisories in dev-only transitive deps (`esbuild` via
`vitest`) and self-hosted-Next concerns whose only remediation is a breaking
upgrade to Next 16. The crate is pinned to the latest patched **14.2.x** line;
the Next-16 jump is intentionally deferred for this prototype.
