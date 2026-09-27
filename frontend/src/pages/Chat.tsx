import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";
import { ApiError, api, describeError } from "../api/client";
import { streamChat } from "../api/stream";
import type {
  Agent,
  Citation,
  ConversationSummary,
  Guardrail,
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
import { CloseIcon, HistoryIcon } from "../components/Icons";

interface Item {
  key: string;
  role: "user" | "assistant";
  text: string;
  agent: Agent | null;
  citations: Citation[];
  sources: Source[];
  verdict?: TurnResult["verdict"];
  guardrail?: Guardrail | null;
  streaming?: boolean;
  error?: string | null;
}

const SUGGESTIONS = [
  "Explain how LoRA works",
  "Critique the baselines of the QLoRA paper",
  "Compare RLHF and DPO",
  "What should I read next?",
];

let keySeq = 0;
const nextKey = () => `m${++keySeq}`;

export function Chat() {
  const { conversationId } = useParams();
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const [items, setItems] = useState<Item[]>([]);
  const [tutor, setTutor] = useState<TutorView | null>(null);
  const [busy, setBusy] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [askOther, setAskOther] = useState(false);
  const [announce, setAnnounce] = useState("");
  const current = useRef<string | undefined>(conversationId);
  const pending = useRef("");
  const frame = useRef<number | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  // Load a conversation opened from history or a link (not one this page just created).
  useEffect(() => {
    if (!conversationId) {
      current.current = undefined;
      setItems([]);
      setTutor(null);
      return;
    }
    if (conversationId === current.current && items.length) return;
    current.current = conversationId;
    setLoadError(null);
    Promise.all([
      api<StoredMessage[]>(`/api/v1/conversations/${conversationId}`),
      api<{ tutor: TutorView | null }>(`/api/v1/conversations/${conversationId}/tutor`),
    ]).then(
      ([messages, state]) => {
        setItems(
          messages.map((m) => ({
            key: nextKey(),
            role: m.role,
            text: m.content,
            agent: m.agent,
            citations: m.citations ?? [],
            sources: [],
          })),
        );
        setTutor(state.tutor);
      },
      (e) => setLoadError(describeError(e)),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end", behavior: "auto" });
  }, [items.length]);

  const patchLast = (patch: Partial<Item> | ((item: Item) => Partial<Item>)) =>
    setItems((prev) => {
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
    async (message: string, action?: TutorAction, extra: { paper_id?: string; equation?: string } = {}) => {
      if (busy) return;
      setBusy(true);
      setAskOther(false);
      setAnnounce("");
      setItems((prev) => [
        ...prev,
        { key: nextKey(), role: "user", text: message, agent: null, citations: [], sources: [] },
        { key: nextKey(), role: "assistant", text: "", agent: null, citations: [], sources: [], streaming: true },
      ]);
      try {
        await streamChat(
          { message, conversation_id: current.current, action, ...extra },
          ({ event, data }) => {
            const d = data as Record<string, unknown>;
            if (event === "meta") {
              const id = d.conversation_id as string;
              if (current.current !== id) {
                current.current = id;
                navigate(`/chat/${id}`, { replace: true });
              }
            } else if (event === "intent") {
              patchLast({ agent: d.agent as Agent });
            } else if (event === "sources") {
              patchLast({ sources: d.sources as Source[] });
            } else if (event === "token") {
              pending.current += d.text as string;
              if (frame.current === null) frame.current = requestAnimationFrame(flushTokens);
            } else if (event === "done") {
              const done = d as unknown as TurnResult;
              if (frame.current !== null) cancelAnimationFrame(frame.current);
              pending.current = "";
              patchLast({
                text: done.answer,
                citations: done.citations,
                verdict: done.verdict,
                guardrail: done.guardrail,
                agent: done.agent,
                streaming: false,
              });
              setTutor(done.tutor);
              setAnnounce("Answer ready");
            } else if (event === "error") {
              patchLast({ streaming: false, error: String(d.message ?? "Something went wrong.") });
            }
          },
        );
      } catch (error) {
        const message =
          error instanceof ApiError && error.code === "DAILY_QUOTA_EXCEEDED"
            ? `${error.message}.`
            : describeError(error);
        patchLast({ streaming: false, error: message });
      } finally {
        patchLast({ streaming: false });
        setBusy(false);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [busy, navigate],
  );

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
        <button className="btn-outline" type="button" onClick={() => navigate("/")}>
          New question
        </button>
      </header>

      {historyOpen && <HistoryPanel onClose={() => setHistoryOpen(false)} />}

      <div className="chat-scroll">
        <div className="chat-column">
          {loadError && (
            <p className="answer-error" role="alert">
              {loadError}
            </p>
          )}
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
                error={item.error}
                hideQuestion={item === lastAssistant ? pendingQuestion : null}
              />
            ),
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

          {pendingQuestion && tutor && !askOther ? (
            <>
              <Exercise tutor={tutor} busy={busy} onSend={(m, a) => send(m, a)} />
              <button className="btn-quiet ask-other" type="button" onClick={() => setAskOther(true)}>
                Ask something else instead
              </button>
            </>
          ) : (
            (items.length > 0 || conversationId) && (
              <Composer busy={busy} onSend={(m) => send(m)} autoFocus={askOther} />
            )
          )}
          <div ref={endRef} />
        </div>
      </div>
      <div className="visually-hidden" aria-live="polite">
        {announce}
      </div>
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  const [recent, setRecent] = useState<ConversationSummary[]>([]);
  const [due, setDue] = useState<MasteryRow[]>([]);
  useEffect(() => {
    api<ConversationSummary[]>("/api/v1/conversations?limit=5").then(setRecent, () => {});
    api<MasteryRow[]>("/api/v1/me/reviews").then(setDue, () => {});
  }, []);
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
          {recent.length ? (
            <ul className="plain-list">
              {recent.map((c) => (
                <li key={c.id}>
                  <Link to={`/chat/${c.id}`}>{c.title || "Untitled conversation"}</Link>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">Your conversations will appear here.</p>
          )}
          {due.length > 0 && (
            <p className="muted">
              <Link to="/reviews">
                {due.length === 1 ? "1 concept is" : `${due.length} concepts are`} due for review
              </Link>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function HistoryPanel({ onClose }: { onClose: () => void }) {
  const [rows, setRows] = useState<ConversationSummary[] | null>(null);
  useEffect(() => {
    api<ConversationSummary[]>("/api/v1/conversations?limit=50").then(setRows, () => setRows([]));
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);
  return (
    <div className="panel-backdrop" onClick={onClose}>
      <aside
        className="panel"
        aria-label="Conversation history"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="panel-head">
          <h2>History</h2>
          <button className="icon-button" type="button" aria-label="Close history" onClick={onClose}>
            <CloseIcon />
          </button>
        </div>
        {rows === null ? (
          <p className="muted">Loading…</p>
        ) : rows.length ? (
          <ul className="plain-list">
            {rows.map((c) => (
              <li key={c.id}>
                <Link to={`/chat/${c.id}`} onClick={onClose}>
                  {c.title || "Untitled conversation"}
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">No conversations yet. Ask a question to start one.</p>
        )}
      </aside>
    </div>
  );
}
