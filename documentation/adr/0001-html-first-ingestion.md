# ADR 0001 — HTML-first ingestion with a column-safe PDF fallback

- **Status:** Accepted (2026-09-21)
- **Code:** `src/pipeline/extractor.py` (tier router), `html_parser.py`, `parser.py` + `layout.py` (PDF), `chunker.py`
- **Tests:** `tests/test_html_parser.py`, `tests/test_extractor.py`, `tests/test_parser.py`, `tests/test_layout.py`, `tests/test_chunker.py`

## Context

The plan (§4, Background Pipeline) asks for "section-aware chunking, LaTeX extraction". ArXivMind is aimed at ML readers, and ML papers are mostly equations. A tutor that quotes a garbled formula teaches the wrong thing.

PDF text extraction loses math. A PDF only has positioned glyphs, so `\frac{QK^\top}{\sqrt{d_k}}` comes back as scattered characters in some reading order. Two-column layouts make it worse, because naive extraction can interleave the two columns.

arXiv publishes an HTML rendering (LaTeXML) for most papers since late 2023. In that HTML every formula is a `<math>` element whose `alttext` attribute holds the author's original LaTeX, sections are real `<section>` elements with ids, and the bibliography is structured.

## Decision

1. **Tier 1: arXiv HTML.** Sections and subsections with anchors. Inline math becomes `$…$` and display math `$$…$$`, taken from `alttext`. We also extract numbered equations, theorems, tables, captions, footnotes and structured references. Result: `math_fidelity = "latex"`.
2. **Tier 2: PDF**, used only when no HTML exists. We use PyMuPDF, and reading order is only corrected when the page is *demonstrably* interleaved. Result: `math_fidelity = "degraded"`.
3. **Provenance on every paper and chunk.** Each one records `parse_method` and `math_fidelity`, and `parse_stats` records why a tier was skipped, so downstream code (the tutor and the guardrail) can refuse to treat degraded math as authoritative.
4. **Formulas are atomic in the chunker.** A chunk boundary never cuts through a `$…$` span. Each chunk carries its section path, and that path is embedded with the text.
5. **Quality gate.** A parse with fewer than `PARSE_MIN_WORDS` words or fewer than `PARSE_MIN_SECTIONS` sections is rejected, so it goes to the dead-letter queue instead of being indexed as junk.

## Measured, not assumed

- **Formulas:** 99.54% of 9,227 unique formulas across 15 real papers were preserved verbatim (HTML tier, compared against the source LaTeX).
- **Reading order:** on 15 papers, compared against the HTML as ground truth, the conservative column logic matches PyMuPDF's stream order and beats `sort=True`, which reorders pages that were already correct.

## Alternatives considered

| Option | Why not |
|---|---|
| PDF only (pdfplumber / PyMuPDF) | Math is lost, and this product is about math. |
| Nougat / Marker (OCR models that emit LaTeX) | They need a GPU to be fast, run for minutes per paper on CPU, and can hallucinate formulas. We already have the real LaTeX from HTML. |
| Download the LaTeX source (`/e-print`) | It is the truest source, but every author's macros, `\input` trees and custom packages must be resolved, which is a compiler-sized problem. HTML is the output of that compiler (LaTeXML). |
| `unstructured` | A heavy dependency (it pulls in detectron/OCR stacks). It solves general documents and knows nothing about arXiv. |
| Always sort PDF blocks by position (`sort=True`) | Measured worse: it breaks single-column pages and equations. |

## Consequences

- Good: most recent papers get exact LaTeX, and older papers still get indexed.
- Good: the fidelity flag lets the answer layer be honest ("this formula came from a PDF and may be garbled").
- Cost: two parsers to maintain, and arXiv's HTML markup may change. The parser tests use saved real HTML so that changes show up as test failures.
- Limitation: papers from before about 2023 often have no HTML, so they get the degraded tier.
