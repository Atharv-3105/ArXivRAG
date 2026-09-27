import { useId, useState } from "react";
import { api, describeError, json, setToken } from "../api/client";

type Mode = "signin" | "register";

export function Login() {
  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const emailId = useId();
  const passwordId = useId();

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const path = mode === "signin" ? "/api/v1/auth/login" : "/api/v1/auth/register";
      const token = await api<{ access_token: string }>(path, {
        method: "POST",
        ...json({ email, password }),
      });
      setToken(token.access_token);
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login">
      <div className="login-column">
        <div className="login-mark" aria-hidden="true">
          a
        </div>
        <h1>ArXivMind</h1>
        <p className="login-lede">
          Learn from arXiv papers with a tutor that cites every claim and asks you one question
          at a time.
        </p>
        <form onSubmit={submit} className="login-form">
          <label htmlFor={emailId}>Email</label>
          <input
            id={emailId}
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <label htmlFor={passwordId}>Password</label>
          <input
            id={passwordId}
            type="password"
            autoComplete={mode === "signin" ? "current-password" : "new-password"}
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          {mode === "register" && <p className="hint">At least 8 characters.</p>}
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
          <button className="btn-primary" type="submit" disabled={busy}>
            {mode === "signin" ? "Sign in" : "Create account"}
          </button>
        </form>
        <p className="login-switch">
          {mode === "signin" ? "New here?" : "Already have an account?"}{" "}
          <button
            className="link-button"
            type="button"
            onClick={() => {
              setMode(mode === "signin" ? "register" : "signin");
              setError(null);
            }}
          >
            {mode === "signin" ? "Create an account" : "Sign in"}
          </button>
        </p>
      </div>
    </div>
  );
}
