// The three honest states of any fetched screen. An empty list and a failed request look nothing
// alike to the person waiting, so they must never share a message.

interface LoadingProps {
  label?: string;
  lines?: number;
}

export function Loading({ label = "Loading", lines = 3 }: LoadingProps) {
  return (
    <div className="skeleton" role="status" aria-live="polite" aria-busy="true">
      <span className="visually-hidden">{label}…</span>
      {Array.from({ length: lines }, (_, i) => (
        <span key={i} className="skeleton-line" style={{ width: `${92 - i * 14}%` }} aria-hidden="true" />
      ))}
    </div>
  );
}

interface ErrorProps {
  message: string;
  onRetry?: () => void;
}

export function ErrorNote({ message, onRetry }: ErrorProps) {
  return (
    <div className="error-note" role="alert">
      <p>{message}</p>
      {onRetry && (
        <button className="btn-outline" type="button" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}
