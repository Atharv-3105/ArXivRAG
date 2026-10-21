import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";
import { api, describeError, friendlyError } from "../api/client";
import { streamChat } from "../api/stream";
import type {
  Agent,
  Citation,
  ConversationSummary,
  Guardrail,
  Intent,
  MasteryRow,
  Source,
  StoredMessage,
  TurnResult,
  TutorAction,
  TutorView,
} from "../api/types";
import { AnswerPage } from "../components/AnswerPage";
import { Composer } from "../components/Composer";
import { Exercise } from "../components/Exercise";
import { ArrowDownIcon, CloseIcon, HistoryIcon, StopIcon } from "../components/Icons";
import { ErrorNote, Loading } from "../components/Status";
import type { TraceInput } from "../lib/trace";
import { useResource } from "../lib/useResource";
import { useTitle } from "../lib/useTitle";

interface Request {
  message: string;
  action?: TutorAction;
  extra: { paper_id?: string; equation?: string };
}

interface Item {
  key: string;
  role: "user" | "assistant";
  text: string;
  agent: Agent | null;
  citations: Citation[];
  sources: Source[];
  intent?: Intent | null;
  concepts?: string[];
  verdict?: TurnResult["verdict"];
  guardrail?: Guardrail | null;
  trace?: TraceInput;
  streaming?: boolean;
  stopped?: boolean;
  error?: string | null;
  request?: Request; // what produced this answer, so a failed one can be retried
}

const SUGGESTIONS = [
  "Explain how LoRA works",
  "Critique the baselines of the QLoRA paper",
  "Compare RLHF and DPO",
  "What should I read next?",
];

// How close to the bottom still counts as "following along" while an answer streams in.
const FOLLOW_PX = 120;

let keySeq = 0;
const nextKey = () => `m${++keySeq}`;

const isAbort = (e: unknown) => e instanceof DOMException && e.name === "AbortError";

export function Chat() {
  const { conversationId } = useParams();
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const [items, setItems] = useState<Item[]>([]);
  const [tutor, setTutor] = useState<TutorView | null>(null);
  const [busy, setBusy] = useState(false);
  const [opening, setOpening] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [askOther, setAskOther] = useState(false);
  const [announce, setAnnounce] = useState("");
  const [away, setAway] = useState(false);
  const current = useRef<string | undefined>(conversationId);
  const pending = useRef("");
  const frame = useRef<number | null>(null);
  const controller = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const follow = useRef(true);
  const endRef = useRef<HTMLDivElement>(null);
  const hasItems = useRef(false);
  hasItems.current = items.length > 0;

  // Load a conversation opened from history or a link (not one this page just created).
  useEffect(() => {
    if (!conversationId) {
      current.current = undefined;
      setItems([]);
      setTutor(null);
      setLoadError(null);
      setOpening(false);
      return;
    }
    if (conversationId === current.current && hasItems.current) return;
    current.current = conversationId;
    setItems([]);
    setTutor(null);
    setLoadError(null);
    setOpening(true);
    const abort = new AbortController();
    Promise.all([
      api<StoredMessage[]>(`/api/v1/conversations/${conversationId}`, { signal: abort.signal }),
      api<{ tutor: TutorView | null }>(`/api/v1/conversations/${conversationId}/tutor`, { signal: abort.signal }),
    ]).then(
      ([messages, state]) => {
        if (abort.signal.aborted) return;
        setItems(
          messages.map((m) => ({
            key: nextKey(),
            role: m.role,
            text: m.content,
            agent: m.agent,
            citations: m.citations ?? [],
            sources: [],
            trace:
              m.role === "assistant" && (m.provider || m.checks)
                ? { agent: m.agent, provider: m.provider, promptVersion: m.prompt_version, checks: m.checks }
                : undefined,
          })),
        );
        setTutor(state.tutor);
        setOpening(false);
        follow.current = true;
      },
      (e) => {
        if (abort.signal.aborted) return;
        setLoadError(describeError(e));
        setOpening(false);
      },
    );
    return () => abort.abort();
  }, [conversationId, loadAttempt]);

  // Follow a streaming answer, unless the reader has scrolled up to re-read something.
  useEffect(() => {
    // Nothing to follow on the landing screen: scrolling to the end there hid the heading on phones.
    if (items.length && follow.current) endRef.current?.scrollIntoView({ block: "end", behavior: "auto" });
  }, [items]);

  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < FOLLOW_PX;
    follow.current = near;
    setAway(!near);
  };
  const jumpToLatest = () => {
    follow.current = true;
    setAway(false);
    endRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  };

  const patchLast = (patch: Partial<Item> | ((item: Item) => Partial<Item>)) =>
    setItems((prev) => {
      if (!prev.length) return prev;
      const next = prev.slice();
      const last = next[next.length - 1];
      next[next.length - 1] = { ...last, ...(typeof patch === "function" ? patch(last) : patch) };
      return next;
    });

  const flushTokens = () => {
    frame.current = null;
    const chunk = pending.current;
    pending.current = "";
    if (chunk) patchLast((item) => ({ text: item.text + chunk }));
  };

  const send = useCallback(
    async (message: string, action?: TutorAction, extra: Request["extra"] = {}) => {
      if (busy) return;
      const request: Request = { message, action, extra };
      const abort = new AbortController();
      controller.current = abort;
      follow.current = true;
      setAway(false);
      setBusy(true);
      setAskOther(false);
      setAnnounce("");
      setItems((prev) => [
        ...prev,
        { key: nextKey(), role: "user", text: message, agent: null, citations: [], sources: [] },
        { key: nextKey(), role: "assistant", text: "", agent: null, citations: [], sources: [], streaming: true, request },
      ]);
      let finished = false;
      try {
        await streamChat(
          { message, conversation_id: current.current, action, ...extra },
          ({ event, data }) => {
            const d = data as Record<string, unknown>;
            if (event === "meta") {
              const id = d.conversation_id as string;
              patchLast({ concepts: (d.concepts as string[]) ?? [] });
              if (current.current !== id) {
                current.current = id;
                navigate(`/chat/${id}`, { replace: true });
              }
            } else if (event === "intent") {
              patchLast({ agent: d.agent as Agent, intent: d as unknown as Intent });
            } else if (event === "sources") {
              patchLast({ sources: d.sources as Source[] });
            } else if (event === "token") {
              pending.current += d.text as string;
              if (frame.current === null) frame.current = requestAnimationFrame(flushTokens);
            } else if (event === "done") {
              finished = true;
              const done = d as unknown as TurnResult;
              if (frame.current !== null) cancelAnimationFrame(frame.current);
              frame.current = null;
              pending.current = "";
              patchLast((item) => ({
                text: done.answer,
                citations: done.citations,
                verdict: done.verdict,
                guardrail: done.guardrail,
                agent: done.agent,
                streaming: false,
                trace: {
                  agent: done.agent,
                  intent: item.intent,
                  sources: item.sources,
                  timings: done.timings,
                  checks: done.checks,
                  guardrail: done.guardrail,
                  provider: done.provider,
                  model: done.model,
                  promptVersion: done.prompt_version,
                  graphStatus: done.graph_status,
                  degraded: done.degraded,
                },
              }));
              setTutor(done.tutor);
              setAnnounce("Answer ready");
            } else if (event === "error") {
              finished = true;
              const text = friendlyError(
                String(d.code ?? ""),
                String(d.message ?? "Something went wrong."),
                typeof d.retry_after === "number" ? d.retry_after : null,
              );
              patchLast({ streaming: false, error: text });
              setAnnounce(text);
            }
          },
          abort.signal,
        );
        if (!finished && !abort.signal.aborted) {
          // The body ended without a "done" or "error" event: the connection was cut mid-answer.
          // Showing the partial text as if it were complete would be wrong.
          const text = "The connection was interrupted before the answer finished.";
          patchLast({ streaming: false, error: text });
          setAnnounce(text);
        }
      } catch (error) {
        if (isAbort(error)) {
          patchLast({ streaming: false, stopped: true });
          setAnnounce("Stopped");
        } else {
          const text = describeError(error);
          patchLast({ streaming: false, error: text });
          setAnnounce(text);
        }
      } finally {
        if (frame.current !== null) cancelAnimationFrame(frame.current);
        frame.current = null;
        pending.current = "";
        patchLast({ streaming: false });
        controller.current = null;
        setBusy(false);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [busy, navigate],
  );

  const stop = () => controller.current?.abort();

  const retry = () => {
    const last = items[items.length - 1];
    if (!last?.request || busy) return;
    const { message, action, extra } = last.request;
    setItems((prev) => prev.slice(0, -2)); // drop the failed pair, send() adds a fresh one
    void send(message, action, extra);
  };

  // Other screens start a turn with ?ask=...&action=...&paper=... (reviews, reading, map).
  useEffect(() => {
    const ask = params.get("ask");
    if (!ask || conversationId) return;
    const action = (params.get("action") as TutorAction | null) ?? undefined;
    const paper_id = params.get("paper") ?? undefined;
    const equation = params.get("equation") ?? undefined;
    setParams({}, { replace: true });
    void send(ask, action, { paper_id, equation });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const lastAssistant = [...items].reverse().find((i) => i.role === "assistant");
  const title = items.find((i) => i.role === "user")?.text;
  const pendingQuestion = !busy ? tutor?.pending_question ?? null : null;
  const reading = tutor?.reading;
  const ready = !opening && !loadError;
  useTitle(title ?? "New question");

  return (
    <div className="chat">
      <header className="chat-header">
        <h1 className="chat-title">{title ?? "New question"}</h1>
        {tutor && tutor.step > 0 && tutor.pending_question && (
          <div className="progress" aria-label={`Question ${tutor.step} of ${tutor.max_steps}`}>
            <span>
              Question {tutor.step} of {tutor.max_steps}
            </span>
            <span className="progress-ticks" aria-hidden="true">
              {Array.from({ length: tutor.max_steps }, (_, i) => (
                <span key={i} className={i < tutor.step ? "tick tick-on" : "tick"} />
              ))}
            </span>
          </div>
        )}
        <button className="btn-outline" type="button" aria-label="History" onClick={() => setHistoryOpen(true)}>
          <HistoryIcon size={18} /> <span className="hide-small">History</span>
        </button>
        {(conversationId || items.length > 0) && (
          <button className="btn-outline" type="button" onClick={() => navigate("/")}>
            New question
          </button>
        )}
      </header>

      {historyOpen && <HistoryPanel onClose={() => setHistoryOpen(false)} />}

      <div className="chat-scroll" ref={scroller} onScroll={onScroll}>
        <div className="chat-column">
          {opening && <Loading label="Opening the conversation" lines={5} />}
          {loadError && <ErrorNote message={loadError} onRetry={() => setLoadAttempt((n) => n + 1)} />}
          {!items.length && !conversationId && <EmptyState onPick={(q) => send(q)} />}
          {items.map((item) =>
            item.role === "user" ? (
              <div className="asked" key={item.key}>
                <span className="asked-mark" aria-hidden="true" />
                <span>{item.text}</span>
              </div>
            ) : (
              <AnswerPage
                key={item.key}
                id={item.key}
                text={item.text}
                agent={item.agent}
                citations={item.citations}
                sources={item.sources}
                verdict={item.verdict}
                guardrail={item.guardrail}
                streaming={item.streaming}
                stopped={item.stopped}
                error={item.error}
                trace={item.trace}
                conversationId={current.current}
                concepts={item.concepts}
                onRetry={item === lastAssistant && item.request ? retry : undefined}
                hideQuestion={item === lastAssistant ? pendingQuestion : null}
              />
            ),
          )}

          {busy && (
            <div className="stopbar">
              <button className="btn-outline" type="button" onClick={stop}>
                <StopIcon size={16} /> Stop
              </button>
            </div>
          )}

          {reading && !busy && (
            <div className="reading-bar">
              <span>
                Section {reading.index + 1} of {reading.total}: {reading.section}
              </span>
              <button className="btn-outline" type="button" onClick={() => send("Next section", "next")}>
                Next section
              </button>
            </div>
          )}

          {ready && pendingQuestion && tutor && !askOther ? (
            <>
              <Exercise tutor={tutor} busy={busy} onSend={(m, a) => send(m, a)} />
              <button className="btn-quiet ask-other" type="button" onClick={() => setAskOther(true)}>
                Ask something else instead
              </button>
            </>
          ) : (
            ready &&
            (items.length > 0 || conversationId) && (
              <Composer busy={busy} onSend={(m) => send(m)} autoFocus={askOther} />
            )
          )}
          <div ref={endRef} />
        </div>
      </div>

      {away && busy && (
        <button className="jump-latest" type="button" onClick={jumpToLatest}>
          <ArrowDownIcon size={16} /> Latest
        </button>
      )}
      <div className="visually-hidden" aria-live="polite">
        {announce}
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  const recent = useResource((signal) => api<ConversationSummary[]>("/api/v1/conversations?limit=5", { signal }));
  const due = useResource((signal) => api<MasteryRow[]>("/api/v1/me/reviews", { signal }));
  const dueCount = due.data?.length ?? 0;
  return (
    <div className="empty">
      <h2 className="empty-title">What do you want to understand?</h2>
      <p className="empty-lede">
        Answers come from the papers in the library, with every claim cited beside the text. When
        you are learning something, the tutor explains one idea and then asks you a question.
      </p>
      <Composer busy={false} onSend={onPick} autoFocus />
      <div className="empty-grid">
        <div>
          <h3>Try</h3>
          <ul className="plain-list">
            {SUGGESTIONS.map((q) => (
              <li key={q}>
                <button className="link-button" type="button" onClick={() => onPick(q)}>
                  {q}
                </button>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3>Pick up where you left off</h3>
          {recent.loading ? (
            <Loading label="Loading your conversations" lines={2} />
          ) : recent.error ? (
            <ErrorNote message={recent.error} onRetry={recent.reload} />
          ) : recent.data?.length ? (
            <ul className="plain-list">
              {recent.data.map((c) => (
                <li key={c.id}>
                  <Link to={`/chat/${c.id}`}>{c.title || "Untitled conversation"}</Link>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">Your conversations will appear here.</p>
          )}
          {dueCount > 0 && (
            <p className="muted">
              <Link to="/reviews">
                {dueCount === 1 ? "1 concept is" : `${dueCount} concepts are`} due for review
              </Link>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function HistoryPanel({ onClose }: { onClose: () => void }) {
  const rows = useResource((signal) => api<ConversationSummary[]>("/api/v1/conversations?limit=50", { signal }));
  const [query, setQuery] = useState("");
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);

  const shown = (rows.data ?? []).filter((c) => (c.title ?? "").toLowerCase().includes(query.trim().toLowerCase()));
  return (
    <div className="panel-backdrop" onClick={onClose}>
      <aside className="panel" role="dialog" aria-modal="true" aria-label="Conversation history" onClick={(e) => e.stopPropagation()}>
        <div className="panel-head">
          <h2>History</h2>
          <button className="icon-button" type="button" aria-label="Close history" onClick={onClose} autoFocus>
            <CloseIcon />
          </button>
        </div>
        {(rows.data?.length ?? 0) > 6 && (
          <input
            className="panel-search"
            type="search"
            value={query}
            placeholder="Search your conversations"
            aria-label="Search your conversations"
            onChange={(e) => setQuery(e.target.value)}
          />
        )}
        {rows.loading ? (
          <Loading label="Loading history" />
        ) : rows.error ? (
          <ErrorNote message={rows.error} onRetry={rows.reload} />
        ) : shown.length ? (
          <ul className="plain-list">
            {shown.map((c) => (
              <li key={c.id}>
                <Link to={`/chat/${c.id}`} onClick={onClose}>
                  {c.title || "Untitled conversation"}
                </Link>
              </li>
            ))}
          </ul>
        ) : query ? (
          <p className="muted">Nothing matches "{query}".</p>
        ) : (
          <p className="muted">No conversations yet. Ask a question to start one.</p>
        )}
      </aside>
    </div>
  );
}
