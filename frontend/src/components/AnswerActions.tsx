import { useState } from "react";
import { api, describeError, json } from "../api/client";
import type { FeedbackSignal } from "../api/types";
import { CopyIcon, ThumbDownIcon, ThumbUpIcon } from "./Icons";

interface Props {
  text: string;
  conversationId?: string;
  concepts?: string[];
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Insecure context or permission denied: fall back to a hidden textarea.
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  }
}

/** Copy, and rate the answer (the rating feeds evaluation, never the learner's mastery). */
export function AnswerActions({ text, conversationId, concepts = [] }: Props) {
  const [copied, setCopied] = useState(false);
  const [sent, setSent] = useState<FeedbackSignal | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const copy = async () => {
    if (await copyText(text)) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    }
  };

  const rate = async (signal: "helpful" | "not_helpful") => {
    if (pending || sent) return;
    setPending(true);
    setError(null);
    try {
      await api("/api/v1/feedback", {
        method: "POST",
        ...json({ signal, concepts, conversation_id: conversationId ?? null }),
      });
      setSent(signal);
    } catch (e) {
      setError(describeError(e));
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="answer-actions">
      <button className="action" type="button" onClick={copy}>
        <CopyIcon size={16} /> {copied ? "Copied" : "Copy"}
      </button>
      {sent ? (
        <span className="action-thanks" role="status">
          Thanks, that helps improve the answers.
        </span>
      ) : (
        <>
          <button className="action" type="button" disabled={pending} onClick={() => rate("helpful")} aria-label="This answer helped">
            <ThumbUpIcon size={16} /> Helpful
          </button>
          <button className="action" type="button" disabled={pending} onClick={() => rate("not_helpful")} aria-label="This answer did not help">
            <ThumbDownIcon size={16} /> Not helpful
          </button>
        </>
      )}
      {error && (
        <span className="action-error" role="alert">
          {error}
        </span>
      )}
    </div>
  );
}
