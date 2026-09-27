import type { Agent } from "../api/types";

// What each specialist's answer is, in the reader's words (not the system's).
export const AGENT_LABEL: Record<Agent, string> = {
  tutor: "Explanation",
  critic: "Critique",
  synthesizer: "Comparison",
  advisor: "Reading advice",
  general: "Answer",
};

export const VERDICT_LABEL = {
  correct: "Correct",
  partial: "Partly right",
  incorrect: "Not yet",
} as const;

// arXiv primary categories -> region names on the map.
export const CATEGORY_NAME: Record<string, string> = {
  "cs.CL": "Language",
  "cs.LG": "Machine learning",
  "cs.CV": "Vision",
  "cs.AI": "Artificial intelligence",
  "stat.ML": "Statistical learning",
  "cs.NE": "Neural computation",
  "cs.IR": "Retrieval",
  "cs.RO": "Robotics",
};

export const categoryName = (c: string | null) => (c ? CATEGORY_NAME[c] ?? c : "Other");

export const year = (iso: string | null | undefined) => (iso ? iso.slice(0, 4) : "");

export const percent = (x: number) => `${Math.round(x * 100)}%`;

export function masteryWords(m: number): string {
  if (m >= 0.7) return "Confident";
  if (m >= 0.35) return "Getting there";
  return "Still building";
}

export function relativeDay(iso: string | null): string {
  if (!iso) return "not scheduled";
  const days = Math.round((new Date(iso).getTime() - Date.now()) / 86_400_000);
  if (days <= 0) return "due now";
  if (days === 1) return "tomorrow";
  return `in ${days} days`;
}
