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
  if (response.status === 401) setToken(null);
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

/** Readable text for an error shown to the user. */
export function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 503 && error.retryAfter)
      return `${error.message} Try again in ${error.retryAfter} seconds.`;
    return error.message;
  }
  return "The server could not be reached. Check your connection and try again.";
}
