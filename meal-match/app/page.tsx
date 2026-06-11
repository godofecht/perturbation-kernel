import Link from "next/link";
import { COOKS, MEALS } from "@/lib/data";
import { MealCard } from "@/components/MealCard";

export default function HomePage() {
  return (
    <>
      <section className="hero">
        <h1>
          Skip the chains. <span className="hl">Match with a home cook</span> and
          the buddies to share the table.
        </h1>
        <p>
          MealMatch pairs you with nearby cooks whose food fits your taste and
          diet — then finds fellow diners to fill the table. Less gig-economy
          delivery, more long dinner with good strangers.
        </p>
        <div className="cta-row">
          <Link href="/match" className="btn primary">
            Find my table →
          </Link>
          <a href="#tonight" className="btn ghost">
            Browse tonight&apos;s cooks
          </a>
        </div>
      </section>

      <section className="section" id="tonight">
        <h2>Cooking tonight</h2>
        <p className="sub">
          {MEALS.length} home-cooked meals from {COOKS.length} cooks across South
          and East London.
        </p>
        <div className="grid">
          {MEALS.map((meal) => (
            <MealCard key={meal.id} meal={meal} />
          ))}
        </div>
      </section>
    </>
  );
}
