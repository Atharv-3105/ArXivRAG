import { useEffect, useId, useState } from "react";
import { useNavigate } from "react-router";
import { api } from "../api/client";
import { year } from "../lib/labels";

interface TopPaper {
  arxiv_id: string;
  title: string;
  published: string | null;
}

const readUrl = (id: string) =>
  `/?ask=${encodeURIComponent(`Walk me through ${id}`)}&action=read&paper=${encodeURIComponent(id)}`;

export function Reading() {
  const [paperId, setPaperId] = useState("");
  const [papers, setPapers] = useState<TopPaper[] | null>(null);
  const navigate = useNavigate();
  const fieldId = useId();

  useEffect(() => {
    api<TopPaper[]>("/api/v1/papers/top?limit=40").then(setPapers, () => setPapers([]));
  }, []);

  return (
    <div className="page-plain">
      <h1 className="page-title">Read a paper with the tutor</h1>
      <p className="page-lede">
        Go through a paper section by section. The tutor names what helps to know first, explains
        each section, and checks your understanding before you move on.
      </p>
      <form
        className="inline-form"
        onSubmit={(e) => {
          e.preventDefault();
          if (paperId.trim()) navigate(readUrl(paperId.trim()));
        }}
      >
        <label htmlFor={fieldId}>arXiv id</label>
        <div className="inline-form-row">
          <input
            id={fieldId}
            value={paperId}
            placeholder="2106.09685"
            onChange={(e) => setPaperId(e.target.value)}
          />
          <button className="btn-primary" type="submit" disabled={!paperId.trim()}>
            Start reading
          </button>
        </div>
      </form>

      <h2 className="section-title">In the library</h2>
      {papers === null ? (
        <p className="muted">Loading…</p>
      ) : papers.length ? (
        <ul className="rows">
          {papers.map((p) => (
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
        <p className="muted">The library is empty. Papers appear here once ingestion has run.</p>
      )}
    </div>
  );
}
