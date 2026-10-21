// Pull an arXiv identifier out of whatever a person pastes: a bare id, a versioned id, an old-style
// id ("hep-th/9901001"), or any arxiv.org abs/pdf/html link.

const NEW_STYLE = String.raw`\d{4}\.\d{4,5}`;
const OLD_STYLE = String.raw`[a-z]+(?:-[a-z]+)*(?:\.[A-Z]{2})?/\d{7}`;
const ID = String.raw`(?:${NEW_STYLE}|${OLD_STYLE})(?:v\d+)?`;

const LINK = new RegExp(String.raw`arxiv\.org/(?:abs|pdf|html)/(${ID})(?:\.pdf)?(?![\w.])`, "i");
const BARE = new RegExp(String.raw`^\s*(?:arxiv:\s*)?(${ID})\s*$`, "i");

/** The id (with its version if one was given), or null if the text is not an arXiv reference. */
export function parseArxivId(input: string): string | null {
  const text = input.trim();
  if (!text) return null;
  const match = BARE.exec(text) ?? LINK.exec(text);
  return match ? match[1] : null;
}
