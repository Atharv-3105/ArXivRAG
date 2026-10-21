import { useId, useState } from "react";
import type { TutorAction, TutorView } from "../api/types";
import { renderInline, stripCitations } from "../lib/render";

interface Props {
  tutor: TutorView;
  busy: boolean;
  onSend: (message: string, action: TutorAction) => void;
}

/** The tutor's one open question, set like a textbook exercise. */
export function Exercise({ tutor, busy, onSend }: Props) {
  const [answer, setAnswer] = useState("");
  const fieldId = useId();
  const hintsLeft = Math.max(0, tutor.max_hints - tutor.hints_used);
  const question = tutor.pending_question ?? "";

  const submit = (event: { preventDefault(): void }) => {
    event.preventDefault();
    if (!answer.trim() || busy) return;
    onSend(answer.trim(), "answer");
    setAnswer("");
  };

  return (
    <section className="exercise" aria-labelledby={`${fieldId}-title`}>
      <h2 id={`${fieldId}-title`}>Exercise {tutor.step || 1}</h2>
      <p className="exercise-q" dangerouslySetInnerHTML={{ __html: renderInline(stripCitations(question)) }} />
      <form onSubmit={submit}>
        <label htmlFor={fieldId}>Your answer</label>
        <textarea
          id={fieldId}
          rows={3}
          value={answer}
          placeholder="Answer in your own words"
          onChange={(e) => setAnswer(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(e);
          }}
        />
        <div className="exercise-actions">
          <button className="btn-primary" type="submit" disabled={busy || !answer.trim()}>
            Check my answer
          </button>
          <button
            className="btn-text"
            type="button"
            disabled={busy || hintsLeft === 0}
            onClick={() => onSend("Hint", "hint")}
          >
            Hint
          </button>
          <button
            className="btn-text"
            type="button"
            disabled={busy}
            onClick={() => onSend("I don't understand", "dont_understand")}
          >
            I don't understand
          </button>
          <button className="btn-quiet" type="button" disabled={busy} onClick={() => onSend("Skip", "skip")}>
            Skip
          </button>
          <span className="exercise-hints">
            {hintsLeft === 1 ? "1 hint left" : `${hintsLeft} hints left`}
          </span>
        </div>
      </form>
    </section>
  );
}
