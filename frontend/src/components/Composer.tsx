import { useId, useState } from "react";

interface Props {
  busy: boolean;
  placeholder?: string;
  onSend: (message: string) => void;
  autoFocus?: boolean;
}

export function Composer({ busy, placeholder, onSend, autoFocus }: Props) {
  const [text, setText] = useState("");
  const id = useId();
  const send = (event: { preventDefault(): void }) => {
    event.preventDefault();
    if (!text.trim() || busy) return;
    onSend(text.trim());
    setText("");
  };
  return (
    <form className="composer" onSubmit={send}>
      <label htmlFor={id} className="visually-hidden">
        Your question
      </label>
      <textarea
        id={id}
        rows={2}
        value={text}
        autoFocus={autoFocus}
        placeholder={placeholder ?? "Ask about a paper, a method or a concept"}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) send(e);
        }}
      />
      <button className="btn-primary" type="submit" disabled={busy || !text.trim()}>
        Ask
      </button>
    </form>
  );
}
