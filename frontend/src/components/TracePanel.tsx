import { useMemo } from "react";
import { buildTrace, formatMs, type TraceInput } from "../lib/trace";
import { CheckIcon, ChevronIcon } from "./Icons";

/** The agent pipeline behind one answer: what ran, what it chose, how long it took, what was checked. */
export function TracePanel({ input }: { input: TraceInput }) {
  const trace = useMemo(() => buildTrace(input), [input]);
  const slowest = Math.max(1, ...trace.steps.map((s) => s.ms ?? 0));

  return (
    <details className="trace">
      <summary>
        <ChevronIcon size={16} />
        <span>How this was answered</span>
        {trace.totalMs !== null && <span className="trace-total">{formatMs(trace.totalMs)}</span>}
      </summary>

      <ol className="trace-steps">
        {trace.steps.map((step) => (
          <li key={step.key} className="trace-step">
            <span className="trace-label">{step.label}</span>
            <span className="trace-detail">{step.detail}</span>
            <span className="trace-ms">{formatMs(step.ms)}</span>
            {step.ms !== null && (
              <span className="trace-bar" aria-hidden="true">
                <span style={{ width: `${Math.max(2, (step.ms / slowest) * 100)}%` }} />
              </span>
            )}
          </li>
        ))}
      </ol>

      {trace.checks.length > 0 && (
        <ul className="trace-checks" aria-label="Output checks">
          {trace.checks.map((c) => (
            <li key={c.label} className={c.ok ? "check-ok" : "check-miss"}>
              {c.ok ? <CheckIcon size={14} /> : <span className="check-x" aria-hidden="true">!</span>}
              <span>{c.label}</span>
              <span className="visually-hidden">{c.ok ? ": passed" : ": not met"}</span>
            </li>
          ))}
        </ul>
      )}

      {trace.facts.length > 0 && (
        <dl className="trace-facts">
          {trace.facts.map((f) => (
            <div key={f.label}>
              <dt>{f.label}</dt>
              <dd>{f.value}</dd>
            </div>
          ))}
        </dl>
      )}

      {trace.notes.length > 0 && <p className="trace-notes">{trace.notes.join(" · ")}</p>}
    </details>
  );
}
