// POST /api/v1/chat/stream reader. EventSource cannot POST or send a bearer token, so the
// Server-Sent Events are parsed from the fetch body stream by hand.

import { authHeaders, toApiError } from "./client";
import type { TutorAction } from "./types";

export interface SSEEvent {
  event: string;
  data: unknown;
}

/** Split a buffer into complete events; the incomplete tail is returned for the next chunk. */
export function parseSSE(buffer: string): { events: SSEEvent[]; rest: string } {
  const normalised = buffer.replace(/\r\n/g, "\n");
  const blocks = normalised.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: SSEEvent[] = [];
  for (const block of blocks) {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (!data.length) continue;
    const raw = data.join("\n");
    let parsed: unknown = raw;
    try {
      parsed = JSON.parse(raw);
    } catch {
      /* plain text data */
    }
    events.push({ event, data: parsed });
  }
  return { events, rest };
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  action?: TutorAction;
  paper_id?: string;
  equation?: string;
}

export async function streamChat(
  request: ChatRequest,
  onEvent: (event: SSEEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/v1/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok || !response.body) throw await toApiError(response);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const { events, rest } = parseSSE(buffer);
    buffer = rest;
    events.forEach(onEvent);
  }
  const { events } = parseSSE(buffer + "\n\n");
  events.forEach(onEvent);
}
