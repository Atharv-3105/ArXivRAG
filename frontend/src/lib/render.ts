// Answer text -> typeset blocks. Each block knows which sources it cites first, so the page can
// set those sources as sidenotes beside it (the Margin design).
//
// Pipeline: protect $...$ / $$...$$ math behind placeholders -> markdown (raw HTML disabled) ->
// split into top-level blocks -> [n] citations to superscript links -> KaTeX back in place.
// Citations are replaced before math returns, so a bracket inside a formula is never touched.

import katex from "katex";
import MarkdownIt from "markdown-it";

const md = new MarkdownIt({ html: false, linkify: true, typographer: true, breaks: false });

const MATH = /(?<!\\)\$\$([\s\S]+?)(?<!\\)\$\$|(?<!\\)\$([^$\n]+?)(?<!\\)\$/g;
const CITE = /(?<![\w\]])\[(\d+(?:\s*[,–-]\s*\d+)*)\](?![(:])/g;

export interface Block {
  html: string;
  cites: number[]; // citation numbers first used in this block, in order
  text: string; // plain text, for matching (e.g. the tutor's question)
}

function expand(group: string): number[] {
  const out: number[] = [];
  for (const part of group.split(/\s*,\s*/)) {
    const [a, b] = part.split(/\s*[–-]\s*/).map(Number);
    if (b !== undefined && b > a && b - a < 20) for (let n = a; n <= b; n++) out.push(n);
    else out.push(a);
  }
  return out.filter((n) => n > 0);
}

export function renderMath(tex: string, display: boolean): string {
  return katex.renderToString(tex, { displayMode: display, throwOnError: false, output: "html" });
}

// Models write maths as \(...\) and \[...\] as often as $...$; markdown would eat the backslash
// ("\[" is an escaped bracket) and the formula would print as raw TeX. Rewrite to the dollar form.
const TEX_DISPLAY = /\\\[([\s\S]+?)\\\]/g;
const TEX_INLINE = /\\\((.+?)\\\)/g;
// 【1】 / 【1†L3-L9】 -> " [1]" (the server normalises new answers; this repairs stored ones).
const FULLWIDTH_CITE = /\s*【(\d+(?:\s*[,–-]\s*\d+)*)(?:†[^】]*)?】/g;

export function normaliseModelText(text: string): string {
  return text
    .replace(TEX_DISPLAY, (_m, tex: string) => `$$${tex}$$`)
    .replace(TEX_INLINE, (_m, tex: string) => `$${tex}$`)
    .replace(FULLWIDTH_CITE, " [$1]");
}

export function renderBlocks(text: string, known?: Set<number>, notePrefix = "note"): Block[] {
  const math: { tex: string; display: boolean }[] = [];
  const protectedText = normaliseModelText(text).replace(MATH, (_m, display: string, inline: string) => {
    math.push({ tex: (display ?? inline).trim(), display: display !== undefined });
    return `@@M${math.length - 1}@@`;
  });

  const tokens = md.parse(protectedText, {});
  const groups: (typeof tokens)[] = [];
  let depth = 0;
  let current: typeof tokens = [];
  for (const token of tokens) {
    current.push(token);
    depth += token.nesting;
    if (depth === 0) {
      groups.push(current);
      current = [];
    }
  }
  if (current.length) groups.push(current);

  const seen = new Set<number>();
  return groups.map((group) => {
    let html = md.renderer.render(group, md.options, {});
    const cites: number[] = [];
    html = html.replace(CITE, (whole, inner: string) => {
      const numbers = expand(inner);
      if (known && !numbers.every((n) => known.has(n))) return whole; // not a citation we have
      for (const n of numbers) {
        if (!seen.has(n)) {
          seen.add(n);
          cites.push(n);
        }
      }
      return numbers
        .map((n) => `<sup class="cite"><a href="#${notePrefix}-${n}" data-cite="${n}" aria-label="Source ${n}">${n}</a></sup>`)
        .join("");
    });
    const plain = group
      .filter((t) => t.type === "inline")
      .map((t) => t.content)
      .join(" ")
      .replace(/@@M(\d+)@@/g, (_m, i: string) => math[Number(i)].tex);
    html = html.replace(/@@M(\d+)@@/g, (_m, i: string) => {
      const item = math[Number(i)];
      return renderMath(item.tex, item.display);
    });
    return { html, cites, text: plain };
  });
}

/** Inline-only rendering (a question, a title): math and emphasis, no block wrapper. */
export function renderInline(text: string): string {
  const math: { tex: string; display: boolean }[] = [];
  const protectedText = text.replace(MATH, (_m, display: string, inline: string) => {
    math.push({ tex: (display ?? inline).trim(), display: false });
    return `@@M${math.length - 1}@@`;
  });
  return md
    .renderInline(protectedText)
    .replace(/@@M(\d+)@@/g, (_m, i: string) => renderMath(math[Number(i)].tex, false));
}

const norm = (s: string) => s.replace(/\s+/g, " ").replace(/[*_`]/g, "").trim().toLowerCase();

/** Drop the block holding the tutor's pending question: the page shows it as the exercise. */
export function withoutQuestion(blocks: Block[], question: string | null): Block[] {
  if (!question || !blocks.length) return blocks;
  const q = norm(question);
  const last = blocks[blocks.length - 1];
  const lastText = norm(last.text);
  if (lastText === q || (q.length > 12 && lastText.endsWith(q))) return blocks.slice(0, -1);
  return blocks;
}
