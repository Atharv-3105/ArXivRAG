import { useId, useState } from "react";
import { useNavigate } from "react-router";
import { ApiError, api, describeError } from "../api/client";
import { ErrorNote, Loading } from "../components/Status";
import { parseArxivId } from "../lib/arxiv";
import { year } from "../lib/labels";
import { useResource } from "../lib/useResource";
import { useTitle } from "../lib/useTitle";

interface TopPaper {
  arxiv_id: string;
  title: string;
  published: string | null;
}

const readUrl = (id: string) =>
  `/?ask=${encodeURIComponent(`Walk me through ${id}`)}&action=read&paper=${encodeURIComponent(id)}`;

export function Reading() {
  const [input, setInput] = useState("");
  const [filter, setFilter] = useState("");
  const [checking, setChecking] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const papers = useResource((signal) => api<TopPaper[]>("/api/v1/papers/top?limit=100", { signal }));
  const navigate = useNavigate();
  const fieldId = useId();
  useTitle("Read a paper");

  // Accept a pasted arXiv link as well as a bare id, and find out now, not three screens later,
  // that the paper is not in the library.
  const start = async (event: React.FormEvent) => {
    event.preventDefault();
    setProblem(null);
    const id = parseArxivId(input);
    if (!id) {
      setProblem("That does not look like an arXiv id. Try 2106.09685, or paste the paper's arXiv link.");
      return;
    }
    setChecking(true);
    try {
      await api(`/api/v1/papers/${encodeURIComponent(id)}`);
      navigate(readUrl(id));
    } catch (e) {
      setProblem(
        e instanceof ApiError && e.status === 404
          ? `arXiv ${id} is not in the library yet, so it cannot be read here. Pick one from the list below.`
          : describeError(e),
      );
    } finally {
      setChecking(false);
    }
  };

  const needle = filter.trim().toLowerCase();
  const shown = (papers.data ?? []).filter(
    (p) => !needle || p.title.toLowerCase().includes(needle) || p.arxiv_id.includes(needle),
  );

  return (
    <div className="page-plain">
      <h1 className="page-title">Read a paper with the tutor</h1>
      <p className="page-lede">
        Go through a paper section by section. The tutor names what helps to know first, explains
        each section, and checks your understanding before you move on.
      </p>
      <form className="inline-form" onSubmit={start} noValidate>
        <label htmlFor={fieldId}>arXiv id or link</label>
        <div className="inline-form-row">
          <input
            id={fieldId}
            value={input}
            placeholder="2106.09685 or https://arxiv.org/abs/2106.09685"
            aria-invalid={problem ? true : undefined}
            aria-describedby={problem ? `${fieldId}-problem` : undefined}
            onChange={(e) => {
              setInput(e.target.value);
              setProblem(null);
            }}
          />
          <button className="btn-primary" type="submit" disabled={!input.trim() || checking}>
            {checking ? "Checking…" : "Start reading"}
          </button>
        </div>
        {problem && (
          <p id={`${fieldId}-problem`} className="form-error" role="alert">
            {problem}
          </p>
        )}
      </form>

      <h2 className="section-title">In the library</h2>
      {papers.loading ? (
        <Loading label="Loading the library" lines={4} />
      ) : papers.error ? (
        <ErrorNote message={papers.error} onRetry={papers.reload} />
      ) : papers.data?.length ? (
        <>
          {papers.data.length > 8 && (
            <input
              className="list-filter"
              type="search"
              value={filter}
              placeholder={`Filter ${papers.data.length} papers by title or id`}
              aria-label="Filter the library"
              onChange={(e) => setFilter(e.target.value)}
            />
          )}
          {shown.length ? (
            <ul className="rows">
              {shown.map((p) => (
                <li key={p.arxiv_id} className="row">
                  <div className="row-main">
                    <div className="row-title serif">{p.title}</div>
                    <div className="row-meta">
                      arXiv {p.arxiv_id}
                      {p.published ? `, ${year(p.published)}` : ""}
                    </div>
                  </div>
                  <button className="btn-outline" type="button" onClick={() => navigate(readUrl(p.arxiv_id))}>
                    Read
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">No paper matches "{filter}".</p>
          )}
        </>
      ) : (
        <p className="muted">The library is empty. Papers appear here once ingestion has run.</p>
      )}
    </div>
  );
}
