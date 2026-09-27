import { select } from "d3-selection";
import { zoom, zoomIdentity, type ZoomBehavior } from "d3-zoom";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router";
import { api, describeError } from "../api/client";
import type { CitingContext, MapEdge, MapNode, PaperDetail, Prerequisite, StudiedPaper } from "../api/types";
import { CloseIcon } from "../components/Icons";
import { categoryName, year } from "../lib/labels";
import { layout, placeLabels, placeName, type Place } from "../lib/terrain";

const W = 1200;
const H = 820;
const baseId = (id: string) => id.replace(/v\d+$/, "");

interface Detail {
  paper: PaperDetail;
  prerequisites: Prerequisite[];
  opinions: CitingContext[];
}

export function MapPage() {
  const [data, setData] = useState<{ nodes: MapNode[]; edges: MapEdge[] } | null>(null);
  const [studied, setStudied] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [transform, setTransform] = useState("translate(0,0) scale(1)");
  const svgRef = useRef<SVGSVGElement>(null);
  const zoomRef = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  useEffect(() => {
    api<{ nodes: MapNode[]; edges: MapEdge[] }>("/api/v1/papers/map?limit=120").then(setData, (e) =>
      setError(describeError(e)),
    );
    api<StudiedPaper[]>("/api/v1/me/papers").then(
      (rows) => setStudied(new Set(rows.map((r) => baseId(r.paper_id)))),
      () => {},
    );
  }, []);

  const terrain = useMemo(() => (data ? layout(data.nodes, data.edges, W, H) : null), [data]);

  useEffect(() => {
    if (!svgRef.current) return;
    const behaviour = zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.6, 4])
      .on("zoom", (event) => setTransform(event.transform.toString()));
    zoomRef.current = behaviour;
    select(svgRef.current).call(behaviour);
  }, [terrain]);

  useEffect(() => {
    if (!selected) return;
    setDetail(null);
    const id = encodeURIComponent(selected);
    Promise.all([
      api<PaperDetail>(`/api/v1/papers/${id}`),
      api<Prerequisite[]>(`/api/v1/papers/${id}/prerequisites?depth=2&limit=6`),
      api<CitingContext[]>(`/api/v1/papers/${id}/citations?function=critical&function=comparative&limit=5`),
    ]).then(
      ([paper, prerequisites, opinions]) => setDetail({ paper, prerequisites, opinions }),
      (e) => setError(describeError(e)),
    );
  }, [selected]);

  const zoomBy = (k: number) => {
    if (svgRef.current && zoomRef.current) select(svgRef.current).call(zoomRef.current.scaleBy, k);
  };
  const resetZoom = () => {
    if (svgRef.current && zoomRef.current) select(svgRef.current).call(zoomRef.current.transform, zoomIdentity);
  };

  const labelled = useMemo(() => {
    if (!terrain) return new Set<string>();
    const top = [...terrain.places].sort((a, b) => b.pagerank - a.pagerank).slice(0, 18);
    return new Set([...top.map((p) => p.id), ...terrain.places.filter((p) => studied.has(baseId(p.id))).map((p) => p.id)]);
  }, [terrain, studied]);

  const isStudied = (p: Place) => studied.has(baseId(p.id));
  const labelText = (p: Place) => `${placeName(p.title)}${p.published ? `, ${year(p.published)}` : ""}`;
  const spots = useMemo(
    () => (terrain ? placeLabels(terrain.places.filter((p) => labelled.has(p.id)), labelText, terrain.places) : new Map()),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [terrain, labelled],
  );

  return (
    <div className="map">
      <header className="map-header">
        <h1>Paper map</h1>
        <p>
          Papers in the library, grouped by field. Higher ground is where influential work
          concentrates; trails show which papers build on which.
        </p>
      </header>
      {error && (
        <p className="form-error map-error" role="alert">
          {error}
        </p>
      )}
      {terrain && terrain.places.length === 0 && (
        <p className="map-empty">The library is empty. Papers appear on the map once ingestion has run.</p>
      )}
      <svg
        ref={svgRef}
        className="map-svg"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        role="group"
        aria-label="Map of papers. Use Tab to move between papers and Enter to open one."
      >
        <g transform={transform}>
          <g className="map-contours" aria-hidden="true">
            {terrain?.contours.map((d, i) => <path key={i} d={d} />)}
          </g>
          <g aria-hidden="true">
            {terrain?.regions.map((r) => (
              <text key={String(r.category)} className="map-region" x={r.x} y={r.y} textAnchor="middle">
                {categoryName(r.category)}
              </text>
            ))}
          </g>
          <g className="map-trails" aria-hidden="true">
            {terrain?.trails.map((t, i) => (
              <line
                key={i}
                className={t.function === "critical" ? "trail trail-critical" : "trail"}
                x1={t.source.x}
                y1={t.source.y}
                x2={t.target.x}
                y2={t.target.y}
              />
            ))}
          </g>
          {terrain?.places.map((p) => (
            <g
              key={p.id}
              className={`place${selected === p.id ? " place-selected" : ""}`}
              transform={`translate(${p.x},${p.y})`}
              role="button"
              tabIndex={0}
              aria-label={`${p.title}, ${year(p.published)}${isStudied(p) ? ", studied" : ""}`}
              onClick={() => setSelected(p.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setSelected(p.id);
                }
              }}
            >
              {selected === p.id && <circle className="place-halo" r={p.r + 9} />}
              <circle className={isStudied(p) ? "place-dot studied" : "place-dot"} r={p.r} />
              {labelled.has(p.id) && (
                <text
                  className="place-label"
                  x={spots.get(p.id)?.dx ?? p.r + 6}
                  y={spots.get(p.id)?.dy ?? 4}
                  textAnchor={spots.get(p.id)?.anchor ?? "start"}
                >
                  {labelText(p)}
                </text>
              )}
            </g>
          ))}
        </g>
      </svg>

      <div className="map-controls">
        <button className="icon-button" type="button" aria-label="Zoom in" onClick={() => zoomBy(1.3)}>
          +
        </button>
        <button className="icon-button" type="button" aria-label="Zoom out" onClick={() => zoomBy(1 / 1.3)}>
          −
        </button>
        <button className="btn-quiet" type="button" onClick={resetZoom}>
          Reset view
        </button>
      </div>

      <div className="map-key" aria-label="Map key">
        <span><i className="key-dot studied" /> You have studied</span>
        <span><i className="key-dot" /> In the library</span>
        <span><i className="key-trail" /> Builds on</span>
        <span><i className="key-trail critical" /> Criticises</span>
      </div>

      {selected && (
        <aside className="sheet" aria-label="Paper details">
          <button className="icon-button sheet-close" type="button" aria-label="Close" onClick={() => setSelected(null)}>
            <CloseIcon />
          </button>
          {!detail ? (
            <p className="muted">Loading…</p>
          ) : (
            <SheetBody detail={detail} studied={studied.has(baseId(detail.paper.arxiv_id))} />
          )}
        </aside>
      )}
    </div>
  );
}

function SheetBody({ detail, studied }: { detail: Detail; studied: boolean }) {
  const { paper, prerequisites, opinions } = detail;
  const read = `/?ask=${encodeURIComponent(`Walk me through ${paper.arxiv_id}`)}&action=read&paper=${encodeURIComponent(paper.arxiv_id)}`;
  const ask = `/?ask=${encodeURIComponent(`Explain the main idea of "${paper.title}"`)}`;
  return (
    <>
      <h2 className="sheet-title">{paper.title}</h2>
      <p className="sheet-meta">
        {paper.authors?.length ? `${paper.authors[0]}${paper.authors.length > 1 ? " et al." : ""}, ` : ""}
        {year(paper.published)},{" "}
        <a href={`https://arxiv.org/abs/${paper.arxiv_id}`} target="_blank" rel="noreferrer">
          arXiv {paper.arxiv_id}
        </a>
      </p>
      <p className="sheet-stats">
        Cited by {paper.cited_by} {paper.cited_by === 1 ? "paper" : "papers"} here
        {paper.pagerank >= 1.5 ? `, ${paper.pagerank.toFixed(1)}× the average influence` : ""}
        {studied ? ". You have studied it." : "."}
      </p>
      {paper.abstract && <p className="sheet-abstract">{paper.abstract}</p>}
      <div className="sheet-actions">
        <Link className="btn-primary map-btn" to={read}>
          Read it with the tutor
        </Link>
        <Link className="btn-outline map-btn" to={ask}>
          Ask about it
        </Link>
      </div>
      {prerequisites.length > 0 && (
        <>
          <h3>Read these first</h3>
          <ul className="plain-list sheet-list">
            {prerequisites.map((p) => (
              <li key={p.arxiv_id}>
                <span className="sheet-paper">{p.title}</span>
                {p.published ? `, ${year(p.published)}` : ""}
              </li>
            ))}
          </ul>
        </>
      )}
      {opinions.length > 0 && (
        <>
          <h3>What other papers say</h3>
          <ul className="plain-list sheet-list">
            {opinions.map((o, i) => (
              <li key={i}>
                <blockquote className="sheet-quote">{o.sentence}</blockquote>
                <span className="sheet-by">
                  {o.citing_title ?? o.citing_id}
                  {o.function === "critical" ? ", critical" : ", comparing"}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </>
  );
}
