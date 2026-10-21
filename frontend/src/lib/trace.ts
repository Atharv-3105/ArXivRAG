// "How this was answered": the agent pipeline behind one answer, in the reader's words.
// Pure data in, display model out, so it can be tested without a DOM.

import type { Agent, Guardrail, Source } from "../api/types";

export interface TraceInput {
  agent: Agent | null;
  intent?: { intent: string; confidence: number; action: string | null } | null;
  sources?: Source[];
  timings?: Record<string, number> | null;
  checks?: Record<string, boolean> | null;
  guardrail?: Guardrail | null;
  provider?: string | null;
  model?: string | null;
  promptVersion?: string | null;
  graphStatus?: string | null;
  degraded?: string | null;
}

export interface TraceStep {
  key: string;
  label: string;
  detail: string;
  ms: number | null;
}

export interface TraceView {
  steps: TraceStep[];
  totalMs: number | null;
  checks: { label: string; ok: boolean }[];
  facts: { label: string; value: string }[];
  notes: string[];
}

const NODE_LABEL: Record<string, string> = {
  route: "Route",
  retrieve: "Retrieve",
  tutor: "Tutor",
  critic: "Critic",
  synthesizer: "Synthesizer",
  advisor: "Advisor",
  general: "Answer",
  guardrail: "Check",
};

export const CHECK_LABEL: Record<string, string> = {
  citations_grounded: "Every citation points at a retrieved source",
  one_question: "Ends with exactly one question",
  trailer_parsed: "Structured data parsed",
  all_sections: "All required sections present",
  comparison_table: "Includes a comparison table",
  two_or_more_papers: "Draws on two or more papers",
  claims_cited: "Every claim is cited",
  has_recommendations: "Contains recommendations",
  guardrail_approved: "Passed the guardrail",
};

export const checkLabel = (key: string) =>
  CHECK_LABEL[key] ?? key.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

const INTENT_WORDS: Record<string, string> = {
  learn: "learning something",
  critique: "critiquing a paper",
  synthesize: "comparing papers",
  recommend: "what to read next",
  chat: "conversation",
};

function modelName(provider?: string | null, model?: string | null): string | null {
  if (!provider || provider === "none") return null;
  if (provider === "advisor") return "Deterministic ranking, no language model";
  const name = (model || "").split("/").pop() || "";
  const host = provider.split(":")[0];
  return name && !name.startsWith(host) ? `${name} via ${host}` : name || host;
}

export function buildTrace(input: TraceInput): TraceView {
  const timings = input.timings ?? {};
  const papers = new Set((input.sources ?? []).map((s) => s.paper_id.replace(/v\d+$/, "")));
  const detail: Record<string, string> = {};

  if (input.intent) {
    const words = INTENT_WORDS[input.intent.intent] ?? input.intent.intent;
    const sure = Math.round(input.intent.confidence * 100);
    detail.route = `Read the question as ${words} (${sure}% sure)${input.intent.action ? `, step: ${input.intent.action}` : ""}`;
  }
  if (input.sources) {
    detail.retrieve = input.sources.length
      ? `${input.sources.length} passages from ${papers.size} ${papers.size === 1 ? "paper" : "papers"}${
          input.graphStatus === "ok" ? ", re-ranked by the citation graph" : ""
        }`
      : "No passages found";
  }
  const writer = input.agent ?? "general";
  const model = modelName(input.provider, input.model);
  detail[writer] = [model, input.promptVersion].filter(Boolean).join(", ");
  if (input.guardrail) {
    const g = input.guardrail;
    detail.guardrail = g.approved
      ? "Approved"
      : `Flagged: ${g.issues.map((i) => i.replace(/_/g, " ")).join(", ") || "see notes"}`;
  }

  const order = ["route", "retrieve", writer, "guardrail"];
  const keys = [...order.filter((k) => k in timings || k in detail), ...Object.keys(timings).filter((k) => !order.includes(k))];
  const steps: TraceStep[] = keys.map((key) => ({
    key,
    label: NODE_LABEL[key] ?? key,
    detail: detail[key] ?? "",
    ms: typeof timings[key] === "number" ? timings[key] : null,
  }));
  const measured = steps.filter((s) => s.ms !== null);
  const totalMs = measured.length ? measured.reduce((sum, s) => sum + (s.ms ?? 0), 0) : null;

  const checks = Object.entries(input.checks ?? {}).map(([key, ok]) => ({ label: checkLabel(key), ok }));

  const facts: TraceView["facts"] = [];
  const g = input.guardrail?.metrics as Record<string, number | boolean> | undefined;
  if (g && !g.skipped && typeof g.sentences === "number") {
    facts.push({ label: "Sentences carrying a citation", value: `${Math.round(Number(g.cited_share ?? 0) * 100)}% of ${g.sentences}` });
    if (Number(g.formulas) > 0) facts.push({ label: "Formulas found in the sources", value: `${g.formulas_grounded} of ${g.formulas}` });
  }

  const notes: string[] = [];
  if (input.degraded) notes.push(`Degraded: ${input.degraded.replace(/_/g, " ")}`);
  for (const issue of input.guardrail?.issues ?? []) notes.push(issue.replace(/_/g, " "));

  return { steps, totalMs, checks, facts, notes };
}

/** "1.2 s", "340 ms", "<1 ms": short enough to sit in a row. */
export function formatMs(ms: number | null): string {
  if (ms === null) return "";
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`;
}
