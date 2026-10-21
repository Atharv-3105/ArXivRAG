import { useEffect, useId, useRef, useState } from "react";
import { isTouchDevice } from "../lib/pointer";

// Mirrors MAX_MESSAGE_CHARS on the server: past it the request is refused, so stop typing here.
export const MESSAGE_LIMIT = 4000;
const COUNTER_FROM = 3500;

interface Props {
  busy: boolean;
  placeholder?: string;
  onSend: (message: string) => void;
  autoFocus?: boolean;
}

export function Composer({ busy, placeholder, onSend, autoFocus }: Props) {
  const [text, setText] = useState("");
  const id = useId();
  const field = useRef<HTMLTextAreaElement>(null);

  // Grow with the text (up to about eight lines) instead of hiding a long question behind a scrollbar.
  useEffect(() => {
    const el = field.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
    el.style.overflowY = el.scrollHeight > 220 ? "auto" : "hidden"; // no scrollbar at rest
  }, [text]);

  const send = (event: { preventDefault(): void }) => {
    event.preventDefault();
    if (!text.trim() || busy) return;
    onSend(text.trim());
    setText("");
  };

  const remaining = MESSAGE_LIMIT - text.length;
  return (
    <form className="composer" onSubmit={send}>
      <label htmlFor={id} className="visually-hidden">
        Your question
      </label>
      <div className="composer-field">
        <textarea
          ref={field}
          id={id}
          rows={2}
          value={text}
          autoFocus={autoFocus && !isTouchDevice()}
          maxLength={MESSAGE_LIMIT}
          placeholder={placeholder ?? "Ask about a paper, a method or a concept"}
          aria-describedby={text.length >= COUNTER_FROM ? `${id}-count` : undefined}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            // Enter confirms a character while an input method (Japanese, Chinese, Korean...) is
            // composing: sending then would submit half a word.
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) send(e);
          }}
        />
        {text.length >= COUNTER_FROM && (
          <span id={`${id}-count`} className={remaining < 100 ? "composer-count composer-count-low" : "composer-count"}>
            {remaining.toLocaleString()} characters left
          </span>
        )}
      </div>
      <button className="btn-primary" type="submit" disabled={busy || !text.trim()}>
        Ask
      </button>
    </form>
  );
}
