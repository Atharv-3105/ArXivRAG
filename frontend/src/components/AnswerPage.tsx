import { useMemo } from "react";
import type { Agent, Citation, Guardrail, Source } from "../api/types";
import { AGENT_LABEL, VERDICT_LABEL, year } from "../lib/labels";
import { renderBlocks, withoutQuestion } from "../lib/render";
import type { TraceInput } from "../lib/trace";
import { AnswerActions } from "./AnswerActions";
import { CheckIcon } from "./Icons";
import { TracePanel } from "./TracePanel";

export interface AnswerProps {
  id: string;
  text: string;
  agent: Agent | null;
  citations: Citation[];
  sources: Source[];
  verdict?: "correct" | "partial" | "incorrect" | null;
  guardrail?: Guardrail | null;
  streaming?: boolean;
  error?: string | null;
  stopped?: boolean;
  hideQuestion?: string | null;
  trace?: TraceInput;
  conversationId?: string;
  concepts?: string[];
  onRetry?: () => void;
}

/** One answer, typeset as a page: each paragraph's sources sit beside it in the margin. */
export function AnswerPage(props: AnswerProps) {
  const { id, text, agent, citations, sources, verdict, guardrail, streaming, error } = props;

  // While streaming, [n] resolves against the retrieved sources; once done, against citations.
  const bySource = useMemo(() => {
    const map = new Map<number, Citation & { published?: string | null }>();
    for (const s of sources) map.set(s.n, s);
    for (const c of citations) map.set(c.n, { ...map.get(c.n), ...c });
    return map;
  }, [sources, citations]);

  const blocks = useMemo(() => {
    const rendered = renderBlocks(text, new Set(bySource.keys()), `note-${id}`);
    return withoutQuestion(rendered, props.hideQuestion ?? null);
  }, [text, bySource, id, props.hideQuestion]);

  return (
    <article className="answer" aria-busy={streaming ? true : undefined}>
      <div className="answer-meta">
        {agent && <span>{verdict ? "Feedback" : AGENT_LABEL[agent]}</span>}
        {verdict && <span className={`verdict verdict-${verdict}`}>{VERDICT_LABEL[verdict]}</span>}
      </div>
      {blocks.map((block, i) => (
        <div className="answer-row" key={i}>
          <div className="answer-block" dangerouslySetInnerHTML={{ __html: block.html }} />
          <aside className="answer-notes" aria-label={block.cites.length ? "Sources" : undefined}>
            {block.cites.map((n) => {
              const src = bySource.get(n);
              if (!src) return null;
              return (
                <div className="note" id={`note-${id}-${n}`} key={n}>
                  <span className="note-n">{n}</span>
                  <div>
                    <a className="note-title" href={src.url} target="_blank" rel="noreferrer">
                      {src.title}
                    </a>
                    <div className="note-meta">
                      {[src.section, year(src.published)].filter(Boolean).join(", ")}
                    </div>
                    <div className="note-id">arXiv {src.paper_id}</div>
                  </div>
                </div>
              );
            })}
          </aside>
        </div>
      ))}
      {streaming && !text && <p className="answer-wait">Reading the sources…</p>}
      {streaming && text && <span className="caret" aria-hidden="true" />}
      {props.stopped && !streaming && (
        <div className="answer-stopped" role="status">
          <span>Stopped. This answer was not saved.</span>
          {props.onRetry && (
            <button className="btn-quiet" type="button" onClick={props.onRetry}>
              Ask again
            </button>
          )}
        </div>
      )}
      {error && (
        <div className="answer-error" role="alert">
          <p>{error}</p>
          {props.onRetry && (
            <button className="btn-outline" type="button" onClick={props.onRetry}>
              Try again
            </button>
          )}
        </div>
      )}
      {!streaming && guardrail?.approved && citations.length > 0 && (
        <p className="grounded">
          <CheckIcon size={16} /> Every claim cites one of these sources
        </p>
      )}
      {!streaming && !error && text && (
        <AnswerActions text={text} conversationId={props.conversationId} concepts={props.concepts} />
      )}
      {!streaming && !error && props.trace && <TracePanel input={props.trace} />}
    </article>
  );
}
