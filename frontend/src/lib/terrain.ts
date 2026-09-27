// The paper map's geography (the Atlas design): papers are places, arXiv categories are regions,
// influence is elevation. A deterministic force layout clusters papers by category and pulls
// citing papers together; contour lines are a kernel density estimate of the papers weighted by
// PageRank, so the "peaks" are where influential work concentrates.

import { contourDensity } from "d3-contour";
import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import { geoPath } from "d3-geo";
import type { MapEdge, MapNode } from "../api/types";

export interface Place extends MapNode, SimulationNodeDatum {
  x: number;
  y: number;
  r: number;
}

export interface Region {
  category: string | null;
  x: number;
  y: number;
  count: number;
}

export interface Terrain {
  places: Place[];
  trails: { source: Place; target: Place; function: MapEdge["function"] }[];
  contours: string[];
  regions: Region[];
}

export const radius = (pagerank: number) => 6 + Math.sqrt(Math.max(pagerank, 0)) * 6;

export function layout(nodes: MapNode[], edges: MapEdge[], width: number, height: number): Terrain {
  const counts = new Map<string | null, number>();
  nodes.forEach((n) => counts.set(n.category, (counts.get(n.category) ?? 0) + 1));
  const categories = [...counts.keys()].sort((a, b) => (counts.get(b)! - counts.get(a)!) || String(a).localeCompare(String(b)));
  const cx = width / 2;
  const cy = height / 2;
  // Regions sit on an ellipse matching the canvas shape, starting on the left, so two fields
  // lie side by side (the canvas is wider than tall) instead of stacked in a narrow strip.
  const spread = categories.length > 1 ? 0.3 : 0;
  const anchor = new Map(
    categories.map((c, i) => {
      const angle = Math.PI + (i / categories.length) * Math.PI * 2;
      return [c, { x: cx + Math.cos(angle) * width * spread, y: cy + Math.sin(angle) * height * spread }] as const;
    }),
  );

  // Deterministic start positions (no Math.random): the same data always draws the same map.
  const places: Place[] = nodes.map((n, i) => {
    const a = anchor.get(n.category)!;
    const angle = i * 2.399963; // golden angle
    const dist = 18 + (i % 7) * 9;
    return { ...n, x: a.x + Math.cos(angle) * dist, y: a.y + Math.sin(angle) * dist, r: radius(n.pagerank) };
  });
  const byId = new Map(places.map((p) => [p.id, p]));
  const links: (SimulationLinkDatum<Place> & { function: MapEdge["function"] })[] = edges
    .filter((e) => byId.has(e.source) && byId.has(e.target))
    .map((e) => ({ source: e.source, target: e.target, function: e.function }));

  const sim = forceSimulation(places)
    .force("link", forceLink<Place, (typeof links)[number]>(links).id((d) => d.id).distance(110).strength(0.15))
    .force("charge", forceManyBody<Place>().strength(-420))
    .force("x", forceX<Place>((d) => anchor.get(d.category)!.x).strength(0.07))
    .force("y", forceY<Place>((d) => anchor.get(d.category)!.y).strength(0.07))
    .force("collide", forceCollide<Place>((d) => d.r + 30))
    .stop();
  for (let i = 0; i < 300; i++) sim.tick();

  const pad = 40;
  places.forEach((p) => {
    p.x = Math.max(pad, Math.min(width - pad, p.x));
    p.y = Math.max(pad, Math.min(height - pad, p.y));
  });

  const density = contourDensity<Place>()
    .x((d) => d.x)
    .y((d) => d.y)
    .weight((d) => 0.4 + d.pagerank)
    .size([width, height])
    .bandwidth(Math.max(28, Math.min(width, height) / 22))
    .thresholds(9)(places);
  const path = geoPath();
  const contours = density.map((c) => path(c) ?? "").filter(Boolean);

  const regions: Region[] = categories.map((category) => {
    const members = places.filter((p) => p.category === category);
    const x = members.reduce((s, p) => s + p.x, 0) / members.length;
    const top = Math.min(...members.map((p) => p.y));
    return { category, x, y: Math.max(28, top - 34), count: members.length };
  });

  const trails = links.map((l) => ({
    source: l.source as Place,
    target: l.target as Place,
    function: l.function,
  }));
  return { places, trails, contours, regions };
}

export interface LabelSpot {
  dx: number;
  dy: number;
  anchor: "start" | "end" | "middle";
}

/** Map-maker's label placement: most influential first, each label takes the first of
 *  right / left / below / above that collides with no label placed before it. */
export function placeLabels(
  places: Place[],
  text: (p: Place) => string,
  markersOf?: Place[],
  charWidth = 7.2,
): Map<string, LabelSpot> {
  const boxes: { x0: number; y0: number; x1: number; y1: number }[] = [];
  const spots = new Map<string, LabelSpot>();
  // Every marker on the map is an obstacle too (except the label's own paper).
  const markers = (markersOf ?? places).map((m) => ({
    id: m.id,
    x0: m.x - m.r - 2,
    y0: m.y - m.r - 2,
    x1: m.x + m.r + 2,
    y1: m.y + m.r + 2,
  }));
  const hits = (b: (typeof boxes)[number], o: (typeof boxes)[number]) =>
    b.x0 < o.x1 && o.x0 < b.x1 && b.y0 < o.y1 && o.y0 < b.y1;
  let current = "";
  const overlaps = (b: (typeof boxes)[number]) =>
    boxes.some((o) => hits(b, o)) || markers.some((m) => m.id !== current && hits(b, m));
  for (const p of [...places].sort((a, b) => b.pagerank - a.pagerank)) {
    current = p.id;
    const w = text(p).length * charWidth;
    const h = 22; // measured glyph box is ~18 units at 14px; keep a little air
    const options: [LabelSpot, (typeof boxes)[number]][] = [
      [{ dx: p.r + 6, dy: 4, anchor: "start" }, { x0: p.x + p.r + 6, y0: p.y - 13, x1: p.x + p.r + 6 + w, y1: p.y - 13 + h }],
      [{ dx: -(p.r + 6), dy: 4, anchor: "end" }, { x0: p.x - p.r - 6 - w, y0: p.y - 13, x1: p.x - p.r - 6, y1: p.y - 13 + h }],
      [{ dx: 0, dy: p.r + 16, anchor: "middle" }, { x0: p.x - w / 2, y0: p.y + p.r + 4, x1: p.x + w / 2, y1: p.y + p.r + 4 + h }],
      [{ dx: 0, dy: -(p.r + 8), anchor: "middle" }, { x0: p.x - w / 2, y0: p.y - p.r - 8 - h, x1: p.x + w / 2, y1: p.y - p.r - 8 }],
    ];
    const [spot, box] = options.find(([, b]) => !overlaps(b)) ?? options[0];
    boxes.push(box);
    spots.set(p.id, spot);
  }
  return spots;
}

/** A short place name: the part of the title before a colon, at most a few words. */
export function placeName(title: string): string {
  const head = title.split(":")[0].trim();
  const words = head.split(/\s+/);
  return words.length > 4 ? words.slice(0, 4).join(" ") + "…" : head;
}
