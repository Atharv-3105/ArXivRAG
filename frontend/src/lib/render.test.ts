import { describe, expect, it } from "vitest";
import { parseSSE } from "../api/stream";
import { renderBlocks, renderInline, stripCitations, withoutQuestion } from "./render";
import { layout, placeLabels, placeName } from "./terrain";

describe("renderBlocks", () => {
  it("turns [n] into sidenote links and records first use per block", () => {
    const blocks = renderBlocks("LoRA freezes W [1].\n\nIt merges back [1] [2].", new Set([1, 2]), "note-x");
    expect(blocks).toHaveLength(2);
    expect(blocks[0].cites).toEqual([1]);
    expect(blocks[1].cites).toEqual([2]); // 1 was already introduced beside the first block
    expect(blocks[0].html).toContain('href="#note-x-1"');
  });

  it("renders LaTeX with KaTeX and never treats brackets inside math as citations", () => {
    const [block] = renderBlocks("The update is $h = W_0 x + [1] x$ here.", new Set([1]));
    expect(block.html).toContain("katex");
    expect(block.cites).toEqual([]);
  });

  it("renders display math as its own block", () => {
    const blocks = renderBlocks("Before.\n\n$$h = W_0 x + BAx$$\n\nAfter.");
    expect(blocks[1].html).toContain("katex-display");
  });

  it("leaves unknown numbers and links alone", () => {
    const [block] = renderBlocks("See [7] and [docs](https://arxiv.org).", new Set([1]));
    expect(block.html).toContain("[7]");
    expect(block.cites).toEqual([]);
  });

  it("escapes raw HTML (no script injection from model output)", () => {
    const [block] = renderBlocks('<img src=x onerror="alert(1)"> text');
    expect(block.html).not.toContain("<img");
  });

  it("expands ranges", () => {
    const [block] = renderBlocks("Several works [1-3].", new Set([1, 2, 3]));
    expect(block.cites).toEqual([1, 2, 3]);
  });
});

describe("withoutQuestion", () => {
  it("drops the final block when it is the pending question", () => {
    const blocks = renderBlocks("Idea [1].\n\nWhy does a low rank suffice?", new Set([1]));
    expect(withoutQuestion(blocks, "Why does a low rank suffice?")).toHaveLength(1);
    expect(withoutQuestion(blocks, "Something else entirely?")).toHaveLength(2);
    expect(withoutQuestion(blocks, null)).toHaveLength(2);
  });
});

describe("renderInline", () => {
  it("renders math inside a question", () => {
    expect(renderInline("Double $r$: what changes?")).toContain("katex");
  });
});

describe("parseSSE", () => {
  it("parses complete events and keeps the incomplete tail", () => {
    const { events, rest } = parseSSE(
      'event: meta\ndata: {"conversation_id": "c1"}\n\nevent: token\ndata: {"text": "Lo"}\n\nevent: tok',
    );
    expect(events).toEqual([
      { event: "meta", data: { conversation_id: "c1" } },
      { event: "token", data: { text: "Lo" } },
    ]);
    expect(rest).toBe("event: tok");
  });

  it("handles CRLF line endings", () => {
    const { events } = parseSSE('event: done\r\ndata: {"answer": "x"}\r\n\r\n');
    expect(events[0].event).toBe("done");
  });
});

describe("terrain", () => {
  const nodes = [
    { id: "a", title: "Attention Is All You Need", published: "2017-06-12", category: "cs.CL", pagerank: 3, in_degree: 5 },
    { id: "b", title: "LoRA: Low-Rank Adaptation", published: "2021-06-17", category: "cs.CL", pagerank: 1, in_degree: 1 },
    { id: "c", title: "Adam", published: "2014-12-22", category: "cs.LG", pagerank: 0.5, in_degree: 1 },
  ];
  const edges = [{ source: "b", target: "a", function: "builds_on" as const }];

  it("is deterministic and stays inside the canvas", () => {
    const one = layout(nodes, edges, 800, 600);
    const two = layout(nodes, edges, 800, 600);
    expect(one.places.map((p) => [p.x, p.y])).toEqual(two.places.map((p) => [p.x, p.y]));
    for (const p of one.places) {
      expect(p.x).toBeGreaterThanOrEqual(40);
      expect(p.x).toBeLessThanOrEqual(760);
    }
    expect(one.trails).toHaveLength(1);
    expect(one.contours.length).toBeGreaterThan(0);
    expect(one.regions.map((r) => r.category)).toEqual(["cs.CL", "cs.LG"]);
  });

  it("places labels without collisions when there is room", () => {
    const places = [
      { ...nodes[0], x: 100, y: 100, r: 10 },
      { ...nodes[1], x: 100, y: 108, r: 8 },
    ];
    const spots = placeLabels(places, () => "A fairly long label");
    expect(spots.get("a")?.anchor).toBe("start"); // most influential keeps the preferred side
    expect(spots.get("b")?.anchor).not.toBe("start"); // the neighbour moves out of its way
  });

  it("handles an empty library", () => {
    expect(layout([], [], 800, 600).places).toEqual([]);
  });

  it("shortens titles to place names", () => {
    expect(placeName("LoRA: Low-Rank Adaptation of Large Language Models")).toBe("LoRA");
    expect(placeName("Training language models to follow instructions")).toBe("Training language models to…");
  });
});

describe("model output quirks", () => {
  it("typesets \\(...\\) and \\[...\\] maths instead of printing raw TeX", () => {
    const inline = renderBlocks("where \\(K_b\\) are the keys").map((b) => b.html).join("");
    expect(inline).toContain("katex");
    expect(inline).not.toContain("\\(");
    const display = renderBlocks("the score is\n\n\\[ s_b = \\alpha \\cdot \\text{mean} \\]").map((b) => b.html).join("");
    expect(display).toContain("katex-display");
  });

  it("links fullwidth 【n】 citations, including line references", () => {
    const blocks = renderBlocks("It prunes blocks【1†L10-L14】 and scales【2】.", new Set([1, 2]));
    expect(blocks[0].cites).toEqual([1, 2]);
    expect(blocks[0].html).not.toContain("【");
  });
});


describe("the exercise question", () => {
  const question =
    "The original purpose of dividing by $\\sqrt{d_k}$ is to prevent dot products from growing too large [1]. What problem does this prevent?";

  it("is hidden from the answer even when it contains maths and a citation marker", () => {
    // Found in the browser: the question appeared twice, once in the answer and once as the exercise.
    const answer = `Scaling matters.\n\n${question}`;
    const blocks = renderBlocks(answer, new Set([1]));
    expect(withoutQuestion(blocks, question)).toHaveLength(blocks.length - 1);
  });

  it("is shown without footnote numbers", () => {
    expect(stripCitations(question)).not.toContain("[1]");
    expect(stripCitations("Why? [2, 3] And [4-6] more.")).toBe("Why? And more.");
    expect(stripCitations("Why is $x[1]$ small [2]?")).toBe("Why is $x[1]$ small?"); // indices in maths stay
  });
});
