import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { api, describeError } from "../api/client";
import type { MasteryRow } from "../api/types";
import { MasteryBar } from "../components/MasteryBar";
import { masteryWords, percent, relativeDay } from "../lib/labels";

export function Reviews() {
  const [due, setDue] = useState<MasteryRow[] | null>(null);
  const [all, setAll] = useState<MasteryRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    Promise.all([api<MasteryRow[]>("/api/v1/me/reviews?limit=50"), api<MasteryRow[]>("/api/v1/me/mastery")]).then(
      ([d, a]) => {
        setDue(d);
        setAll(a);
      },
      (e) => setError(describeError(e)),
    );
  }, []);

  const review = (row: MasteryRow) =>
    navigate(`/?ask=${encodeURIComponent(`Quiz me on ${row.label}`)}&action=review`);

  return (
    <div className="page-plain">
      <h1 className="page-title">Reviews</h1>
      <p className="page-lede">
        Short recall questions on concepts you studied, spaced out so they stick: each right answer
        pushes the next review further away.
      </p>
      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}
      <h2 className="section-title">Due now</h2>
      {due === null ? (
        <p className="muted">Loading…</p>
      ) : due.length ? (
        <ul className="rows">
          {due.map((row) => (
            <li key={row.concept} className="row">
              <div className="row-main">
                <div className="row-title">{row.label}</div>
                <div className="row-meta">
                  {masteryWords(row.mastery)}, {percent(row.mastery)}
                </div>
              </div>
              <MasteryBar value={row.mastery} label={`Grasp of ${row.label}`} />
              <button className="btn-primary" type="button" onClick={() => review(row)}>
                Review now
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">Nothing is due. Reviews appear here a day after you study a concept.</p>
      )}

      <h2 className="section-title">Everything you have studied</h2>
      {all.length ? (
        <ul className="rows">
          {all.map((row) => (
            <li key={row.concept} className="row">
              <div className="row-main">
                <div className="row-title">{row.label}</div>
                <div className="row-meta">
                  {percent(row.mastery)}, next review {relativeDay(row.next_review_at)}
                  {row.locked ? ", set by you" : ""}
                </div>
              </div>
              <MasteryBar value={row.mastery} label={`Grasp of ${row.label}`} />
              <button className="btn-outline" type="button" onClick={() => review(row)}>
                Practise
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">Ask the tutor to explain a concept, and it will appear here.</p>
      )}
    </div>
  );
}
