import { useNavigate } from "react-router";
import { api } from "../api/client";
import type { MasteryRow } from "../api/types";
import { MasteryBar } from "../components/MasteryBar";
import { ErrorNote, Loading } from "../components/Status";
import { masteryWords, percent, relativeDay } from "../lib/labels";
import { useResource } from "../lib/useResource";
import { useTitle } from "../lib/useTitle";

export function Reviews() {
  const due = useResource((signal) => api<MasteryRow[]>("/api/v1/me/reviews?limit=50", { signal }));
  const all = useResource((signal) => api<MasteryRow[]>("/api/v1/me/mastery", { signal }));
  const navigate = useNavigate();
  useTitle("Reviews");

  const review = (row: MasteryRow) =>
    navigate(`/?ask=${encodeURIComponent(`Quiz me on ${row.label}`)}&action=review`);

  return (
    <div className="page-plain">
      <h1 className="page-title">Reviews</h1>
      <p className="page-lede">
        Short recall questions on concepts you studied, spaced out so they stick: each right answer
        pushes the next review further away.
      </p>

      <h2 className="section-title">Due now</h2>
      {due.loading ? (
        <Loading label="Loading what is due" lines={2} />
      ) : due.error ? (
        <ErrorNote message={due.error} onRetry={due.reload} />
      ) : due.data?.length ? (
        <ul className="rows">
          {due.data.map((row) => (
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
      {all.loading ? (
        <Loading label="Loading your concepts" lines={3} />
      ) : all.error ? (
        <ErrorNote message={all.error} onRetry={all.reload} />
      ) : all.data?.length ? (
        <ul className="rows">
          {all.data.map((row) => (
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
