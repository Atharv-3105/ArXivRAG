// Thin fetch wrapper: bearer token, JSON, and the API's one error shape.

const TOKEN_KEY = "arxivmind.token";

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public retryAfter: number | null = null,
  ) {
    super(message);
  }
}

// The token lives in localStorage so a reload keeps you signed in. The trade-off (readable by
// injected script) is contained by the strict Content-Security-Policy the API sends with the app.
export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* private mode: the session lasts until the tab closes */
  }
  window.dispatchEvent(new Event("auth:change"));
}

// Where the person was when their session ended (a rejected token), so the sign-in page can say
// why they are there and send them back. sessionStorage: it should not outlive the tab.
const EXPIRED_KEY = "arxivmind.expired";

export function rememberExpired(path: string): void {
  try {
    sessionStorage.setItem(EXPIRED_KEY, path);
  } catch {
    /* storage unavailable: the notice is skipped, nothing else breaks */
  }
}

/** The path the ended session was on, or null. Leaves it in place until `takeExpired`. */
export function peekExpired(): string | null {
  try {
    return sessionStorage.getItem(EXPIRED_KEY);
  } catch {
    return null;
  }
}

export function takeExpired(): string | null {
  const path = peekExpired();
  try {
    sessionStorage.removeItem(EXPIRED_KEY);
  } catch {
    /* ignore */
  }
  // Only ever go back to a page of this app, never to a path someone planted in storage.
  return path && path.startsWith("/") && !path.startsWith("//") && !path.startsWith("/login") ? path : null;
}

export function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function toApiError(response: Response): Promise<ApiError> {
  let code = "HTTP_" + response.status;
  let message = response.statusText || "Request failed";
  try {
    const body = await response.json();
    if (body?.error) {
      code = body.error.code ?? code;
      message = body.error.message ?? message;
    }
  } catch {
    /* not JSON */
  }
  const retry = response.headers.get("retry-after");
  // A 401 while holding a token means the session ended. (A wrong password at sign-in has no token.)
  if (response.status === 401 && getToken()) {
    rememberExpired(window.location.pathname + window.location.search);
    setToken(null);
  }
  return new ApiError(response.status, code, message, retry ? Number(retry) : null);
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...authHeaders(),
      ...(init.headers ?? {}),
    },
  });
  if (!response.ok) throw await toApiError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const json = (body: unknown): RequestInit => ({ body: JSON.stringify(body) });

/**
 * What to tell a person about a failure. The server's messages are written for logs ("No LLM
 * provider could serve the request. Tried: gemini (rate_limited), ..."), so known codes get plain
 * wording and the unknown ones fall back to the server's text.
 */
/** "45 seconds", "about 15 minutes": a wait a person can plan around. */
export function humanWait(seconds: number): string {
  if (seconds < 90) return `${Math.max(1, Math.round(seconds))} seconds`;
  const minutes = Math.round(seconds / 60);
  return minutes < 90 ? `about ${minutes} minutes` : `about ${Math.round(minutes / 60)} hours`;
}

export function friendlyError(code: string, message: string, retryAfter: number | null = null): string {
  const later = retryAfter ? ` Try again in ${humanWait(retryAfter)}.` : " Try again in a minute.";
  switch (code) {
    case "LLM_UNAVAILABLE":
      return `The language models are busy or over their free quota right now.${later}`;
    case "RATE_LIMITED":
      // Two kinds: asking too fast (chat), and too many sign-in or sign-up attempts.
      if (/sign-in|account/i.test(message)) {
        return `${message.replace(/\s*Try again later\.?$/i, "").replace(/\.?$/, ".")}${later}`;
      }
      return `You are asking quickly.${retryAfter ? ` Try again in ${humanWait(retryAfter)}.` : " Wait a moment and try again."}`;
    case "DAILY_QUOTA_EXCEEDED":
      return "You have reached today's question limit. It resets tomorrow.";
    case "VALIDATION_ERROR":
      return "That could not be sent. Check the text and try again.";
    case "INTERNAL_ERROR":
      return "Something went wrong on our side. Try again.";
    case "REGISTRATION_CLOSED":
      return "Sign-ups are closed right now.";
    case "INVALID_CREDENTIALS":
      return "That email and password do not match an account.";
    case "EMAIL_TAKEN":
    case "USER_EXISTS":
      return "An account with that email already exists. Sign in instead.";
    default:
      return message;
  }
}

/** Readable text for an error shown to the user. */
export function describeError(error: unknown): string {
  if (error instanceof ApiError) return friendlyError(error.code, error.message, error.retryAfter);
  if (error instanceof DOMException && error.name === "AbortError") return "Cancelled.";
  return "The server could not be reached. Check your connection and try again.";
}
