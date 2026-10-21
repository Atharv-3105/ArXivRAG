import { useEffect, useId, useState } from "react";
import { api, describeError, json, setToken } from "../api/client";
import type { MasteryRow, Me as MeData } from "../api/types";
import { MasteryBar } from "../components/MasteryBar";
import { percent } from "../lib/labels";
import { applyTheme, getTheme, setTheme, type Theme } from "../lib/theme";
import { useTitle } from "../lib/useTitle";

const LEVELS = [
  ["beginner", "New to the field: define terms, use analogies"],
  ["intermediate", "Comfortable with ML basics"],
  ["advanced", "Expert: be dense and technical"],
] as const;
const MATH = [
  ["light", "Light: mostly words"],
  ["standard", "Standard: formulas where they help"],
  ["rigorous", "Rigorous: full mathematical detail"],
] as const;

export function Me() {
  const [me, setMe] = useState<MeData | null>(null);
  const [mastery, setMastery] = useState<MasteryRow[]>([]);
  const [goals, setGoals] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [password, setPassword] = useState("");
  const [theme, setThemeState] = useState<Theme>(getTheme);
  useTitle("You");
  const ids = { level: useId(), math: useId(), goals: useId(), password: useId(), theme: useId() };

  // A confirmation that stays forever stops being read.
  useEffect(() => {
    if (!status) return;
    const t = window.setTimeout(() => setStatus(null), 4000);
    return () => window.clearTimeout(t);
  }, [status]);

  const load = () =>
    Promise.all([api<MeData>("/api/v1/me"), api<MasteryRow[]>("/api/v1/me/mastery")]).then(
      ([m, rows]) => {
        setMe(m);
        setGoals(m.profile.goals.join("\n"));
        setMastery(rows);
      },
      (e) => setError(describeError(e)),
    );

  useEffect(() => {
    void load();
  }, []);

  const patch = async (body: Record<string, unknown>, done: string) => {
    setError(null);
    try {
      setMe(await api<MeData>("/api/v1/me/profile", { method: "PATCH", ...json(body) }));
      setStatus(done);
    } catch (e) {
      setError(describeError(e));
    }
  };

  const setConcept = async (row: MasteryRow, value: number | null) => {
    try {
      await api(`/api/v1/me/mastery/${row.concept}`, { method: "PUT", ...json({ mastery: value }) });
      setMastery(await api<MasteryRow[]>("/api/v1/me/mastery"));
    } catch (e) {
      setError(describeError(e));
    }
  };

  const exportData = async () => {
    setError(null);
    try {
      const data = await api<unknown>("/api/v1/me/export");
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = "arxivmind-export.json";
      a.click();
      URL.revokeObjectURL(url);
      setStatus("Your data was downloaded");
    } catch (e) {
      setError(describeError(e));
    }
  };

  const deleteAccount = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!window.confirm("Delete your account, history and progress? This cannot be undone.")) return;
    setError(null);
    try {
      await api("/api/v1/me", { method: "DELETE", ...json({ password }) });
      setToken(null);
    } catch (e) {
      setError(describeError(e));
    }
  };

  if (!me) return <div className="page-plain">{error ? <p className="form-error">{error}</p> : <p className="muted">Loading…</p>}</div>;
  const locked = (field: string) => me.locked_fields.includes(field);

  return (
    <div className="page-plain">
      <h1 className="page-title">You</h1>
      <p className="page-lede">{me.email}</p>
      {status && (
        <p className="form-ok" role="status">
          {status}
        </p>
      )}
      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}

      <h2 className="section-title">How answers are pitched</h2>
      <div className="field">
        <label htmlFor={ids.level}>Your level</label>
        <select
          id={ids.level}
          value={me.profile.expertise_level}
          onChange={(e) => patch({ expertise_level: e.target.value }, "Level saved")}
        >
          {LEVELS.map(([v, text]) => (
            <option key={v} value={v}>
              {text}
            </option>
          ))}
        </select>
        <p className="hint">
          {locked("expertise_level") ? (
            <>
              You set this, so ArXivMind will not change it.{" "}
              <button className="link-button" type="button" onClick={() => patch({ unlock: ["expertise_level"] }, "Your level will now follow your answers")}>
                Let it follow my answers
              </button>
            </>
          ) : (
            "Adjusted automatically from your answers until you choose one."
          )}
        </p>
      </div>
      <div className="field">
        <label htmlFor={ids.math}>Maths in answers</label>
        <select
          id={ids.math}
          value={me.profile.math_depth}
          onChange={(e) => patch({ math_depth: e.target.value }, "Maths preference saved")}
        >
          {MATH.map(([v, text]) => (
            <option key={v} value={v}>
              {text}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <label htmlFor={ids.goals}>What you are working towards</label>
        <textarea
          id={ids.goals}
          rows={3}
          value={goals}
          placeholder="One goal per line, for example: understand RLHF well enough to implement it"
          onChange={(e) => setGoals(e.target.value)}
        />
        <button
          className="btn-outline"
          type="button"
          onClick={() =>
            patch({ goals: goals.split("\n").map((g) => g.trim()).filter(Boolean).slice(0, 10) }, "Goals saved")
          }
        >
          Save goals
        </button>
      </div>

      <h2 className="section-title">Appearance</h2>
      <div className="field">
        <label htmlFor={ids.theme}>Theme</label>
        <select
          id={ids.theme}
          value={theme}
          onChange={(e) => {
            const next = e.target.value as Theme;
            setThemeState(next);
            setTheme(next);
            applyTheme(next);
          }}
        >
          <option value="system">Follow my device</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </div>

      <h2 className="section-title">What you know</h2>
      {mastery.length ? (
        <ul className="rows">
          {mastery.map((row) => (
            <li className="row" key={row.concept}>
              <div className="row-main">
                <div className="row-title">{row.label}</div>
                <div className="row-meta">
                  {percent(row.mastery)}
                  {row.locked ? ", set by you" : `, from ${row.evidence} answers and questions`}
                </div>
              </div>
              <MasteryBar value={row.mastery} label={`Grasp of ${row.label}`} />
              {row.locked ? (
                <button className="btn-outline" type="button" onClick={() => setConcept(row, null)}>
                  Track it again
                </button>
              ) : (
                <button className="btn-outline" type="button" onClick={() => setConcept(row, 0.9)}>
                  I know this
                </button>
              )}
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">Nothing yet. Concepts appear as you ask about them and answer the tutor.</p>
      )}

      <h2 className="section-title">Your data</h2>
      <p className="page-text">
        Download everything ArXivMind stores about you: profile, progress, history and the audit
        trail of every change to your profile.
      </p>
      <button className="btn-outline" type="button" onClick={exportData}>
        Download my data
      </button>
      <form className="danger-zone" onSubmit={deleteAccount}>
        <h3>Delete account</h3>
        <p className="page-text">This erases your account, history and progress. It cannot be undone.</p>
        <label htmlFor={ids.password}>Confirm with your password</label>
        <div className="inline-form-row">
          <input
            id={ids.password}
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <button className="btn-danger" type="submit" disabled={!password}>
            Delete account
          </button>
        </div>
      </form>
      <button className="btn-quiet sign-out" type="button" onClick={() => setToken(null)}>
        Sign out
      </button>
    </div>
  );
}
