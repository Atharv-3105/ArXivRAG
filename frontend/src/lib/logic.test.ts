import { describe, expect, it } from "vitest";
import { friendlyError, humanWait } from "../api/client";
import { parseArxivId } from "./arxiv";
import { buildTrace, checkLabel, formatMs } from "./trace";
import type { Source } from "../api/types";

describe("parseArxivId", () => {
  it.each([
    ["2106.09685", "2106.09685"],
    ["  2106.09685v2  ", "2106.09685v2"],
    ["arXiv:2106.09685", "2106.09685"],
    ["https://arxiv.org/abs/2106.09685", "2106.09685"],
    ["https://arxiv.org/abs/2106.09685v3", "2106.09685v3"],
    ["https://arxiv.org/pdf/2106.09685v1.pdf", "2106.09685v1"],
    ["https://arxiv.org/html/2305.14314", "2305.14314"],
    ["hep-th/9901001", "hep-th/9901001"],
    ["https://arxiv.org/abs/cs.LG/0703124", "cs.LG/0703124"],
    ["0704.0001", "0704.0001"], // four-digit sequence numbers were used before 2015
  ])("%s -> %s", (input, expected) => {
    expect(parseArxivId(input)).toBe(expected);
  });

  it("rejects things that are not arXiv references", () => {
    for (const text of ["", "   ", "LoRA", "hello world", "https://example.com/abs/2106.09685", "12345.6789012", "2401.123"]) {
      expect(parseArxivId(text), text).toBeNull();
    }
  });
});

const source = (paper_id: string, n: number): Source =>
  ({ n, paper_id, title: "T", section: "S", url: "u", chunk_id: String(n), content: "", score: 1, math_fidelity: "latex", role: "retrieved" }) as Source;

describe("buildTrace", () => {
  const base = {
    agent: "tutor" as const,
    intent: { intent: "learn", confidence: 1, action: "explain" },
    sources: [source("2106.09685v2", 1), source("2106.09685v2", 2), source("2305.14314v1", 3)],
    timings: { route: 1.2, retrieve: 1.6, tutor: 3560.4, guardrail: 1.8 },
    checks: { citations_grounded: true, one_question: false },
    guardrail: { approved: true, confidence: 1, issues: [], metrics: { sentences: 9, cited_share: 0.889, formulas: 18, formulas_grounded: 6 } },
    provider: "gemini",
    model: "gemini-2.5-flash",
    promptVersion: "tutor_explain@v2",
    graphStatus: "ok",
  };

  it("orders the pipeline route, retrieve, writer, check and names what each did", () => {
    const t = buildTrace(base);
    expect(t.steps.map((s) => s.key)).toEqual(["route", "retrieve", "tutor", "guardrail"]);
    expect(t.steps[0].detail).toContain("learning something");
    expect(t.steps[1].detail).toBe("3 passages from 2 papers, re-ranked by the citation graph");
    expect(t.steps[2].detail).toBe("gemini-2.5-flash, tutor_explain@v2");
    expect(t.steps[3].detail).toBe("Approved");
  });

  it("totals the measured time and keeps each step's own", () => {
    const t = buildTrace(base);
    expect(t.totalMs).toBeCloseTo(3564.99, 1);
    expect(t.steps[2].ms).toBe(3560.4);
  });

  it("lists checks with readable labels and their outcome", () => {
    const t = buildTrace(base);
    expect(t.checks).toEqual([
      { label: "Every citation points at a retrieved source", ok: true },
      { label: "Ends with exactly one question", ok: false },
    ]);
  });

  it("reports the guardrail's own measurements", () => {
    const t = buildTrace(base);
    expect(t.facts).toContainEqual({ label: "Sentences carrying a citation", value: "89% of 9" });
    expect(t.facts).toContainEqual({ label: "Formulas found in the sources", value: "6 of 18" });
  });

  it("names the host when the model is served through a router", () => {
    const t = buildTrace({ agent: "tutor", provider: "openrouter:nvidia/nemotron-3-super:free", model: "nvidia/nemotron-3-super:free" });
    expect(t.steps[0].detail).toBe("nemotron-3-super:free via openrouter");
  });

  it("says plainly when no language model was involved", () => {
    const t = buildTrace({ agent: "advisor", provider: "advisor", model: "deterministic-ranker", timings: { advisor: 2.2 } });
    expect(t.steps[0].detail).toContain("no language model");
  });

  it("surfaces a flagged answer and a degraded turn", () => {
    const t = buildTrace({
      agent: "tutor",
      guardrail: { approved: false, confidence: 0.4, issues: ["uncited_claims"], metrics: {} },
      degraded: "no_sources",
      timings: { guardrail: 1 },
    });
    expect(t.steps.at(-1)?.detail).toBe("Flagged: uncited claims");
    expect(t.notes).toEqual(["Degraded: no sources", "uncited claims"]);
  });

  it("describes retrieval from the stored summary when the sources themselves are gone", () => {
    const t = buildTrace({ agent: "tutor", retrieval: { passages: 10, papers: 9, graph: "ok" }, timings: { retrieve: 2 } });
    expect(t.steps.find((s) => s.key === "retrieve")?.detail).toBe("10 passages from 9 papers, re-ranked by the citation graph");
  });

  it("survives a stored message that has almost nothing", () => {
    const t = buildTrace({ agent: null });
    expect(t.steps.length).toBe(1); // just the writer
    expect(t.totalMs).toBeNull();
  });
});

describe("formatting", () => {
  it("formats durations compactly", () => {
    expect(formatMs(0.4)).toBe("<1 ms");
    expect(formatMs(340)).toBe("340 ms");
    expect(formatMs(1234)).toBe("1.2 s");
    expect(formatMs(33000)).toBe("33 s");
    expect(formatMs(null)).toBe("");
  });

  it("falls back to a readable label for an unknown check", () => {
    expect(checkLabel("some_new_check")).toBe("Some new check");
  });
});


describe("friendlyError", () => {
  it("never shows the server's log-style text for a model outage", () => {
    const text = friendlyError("LLM_UNAVAILABLE", "No LLM provider could serve the request. Tried: gemini (rate_limited)", 42);
    expect(text).not.toContain("Tried:");
    expect(text).toContain("42 seconds");
  });

  it("tells a chat rate limit from a sign-in lockout and gives the wait in sensible units", () => {
    expect(friendlyError("RATE_LIMITED", "Too many requests, slow down", 20)).toBe("You are asking quickly. Try again in 20 seconds.");
    expect(friendlyError("RATE_LIMITED", "Too many failed sign-in attempts. Try again later.", 900)).toBe(
      "Too many failed sign-in attempts. Try again in about 15 minutes.",
    );
    expect(friendlyError("RATE_LIMITED", "Too many accounts created from this address. Try again later.", 3600)).toContain("about 60 minutes");
  });

  it("explains a refused simultaneous turn", () => {
    expect(friendlyError("TURN_IN_PROGRESS", "A reply is still being written")).toContain("Wait for it to finish");
  });

  it("falls back to the server's message for codes it does not know", () => {
    expect(friendlyError("SOMETHING_NEW", "Specific server text")).toBe("Specific server text");
  });

  it("formats waits", () => {
    expect(humanWait(5)).toBe("5 seconds");
    expect(humanWait(120)).toBe("about 2 minutes");
    expect(humanWait(7200)).toBe("about 2 hours");
  });
});
