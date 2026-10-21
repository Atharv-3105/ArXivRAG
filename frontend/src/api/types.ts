// Mirrors the API's response models (src/api/routes/*.py). documentation/api/openapi.json is the source.

export type Agent = "tutor" | "critic" | "synthesizer" | "advisor" | "general";
export type TutorAction =
  | "answer"
  | "hint"
  | "dont_understand"
  | "skip"
  | "review"
  | "read"
  | "next"
  | "equation";

export interface Citation {
  n: number;
  paper_id: string;
  title: string;
  section: string;
  url: string;
}

export interface Source extends Citation {
  chunk_id: string;
  content: string;
  score: number;
  math_fidelity: string;
  role: string;
  published?: string | null;
}

export interface TutorView {
  mode: string | null;
  topic: string | null;
  concept: string | null;
  step: number;
  max_steps: number;
  level: number;
  hints_used: number;
  max_hints: number;
  pending_question: string | null;
  reading: { paper_id: string; title: string; index: number; total: number; section: string } | null;
}

export interface Guardrail {
  approved: boolean;
  confidence: number;
  issues: string[];
  metrics: Record<string, unknown>;
}

export interface Intent {
  intent: string;
  confidence: number;
  agent: Agent;
  action: string | null;
}

export interface TurnResult {
  answer: string;
  citations: Citation[];
  ungrounded: number[];
  degraded: string | null;
  graph_status?: string;
  provider: string;
  model: string;
  agent: Agent | null;
  prompt_version: string | null;
  checks: Record<string, boolean>;
  timings?: Record<string, number>;
  verdict: "correct" | "partial" | "incorrect" | null;
  tutor: TutorView | null;
  guardrail: Guardrail | null;
}

export type FeedbackSignal = "helpful" | "not_helpful" | "understood" | "confused";

export interface StoredMessage {
  role: "user" | "assistant";
  content: string;
  intent: string | null;
  citations: Citation[] | null;
  agent: Agent | null;
  provider?: string | null;
  prompt_version?: string | null;
  checks?: Record<string, boolean> | null;
  at: string;
}

export interface ConversationSummary {
  id: string;
  title: string | null;
  updated_at?: string;
}

export interface Me {
  user_id: number;
  email: string;
  display_name: string | null;
  is_admin: boolean;
  profile: {
    expertise_level: "beginner" | "intermediate" | "advanced";
    math_depth: "light" | "standard" | "rigorous";
    goals: string[];
    interests: string[];
    inferred_interests: Record<string, number>;
  };
  locked_fields: string[];
}

export interface MasteryRow {
  concept: string;
  label: string;
  mastery: number;
  stored: number;
  evidence: number;
  locked: boolean;
  next_review_at: string | null;
}

export interface MapNode {
  id: string;
  title: string;
  published: string | null;
  category: string | null;
  pagerank: number;
  in_degree: number;
}

export interface MapEdge {
  source: string;
  target: string;
  function: "critical" | "comparative" | "builds_on" | "neutral";
}

export interface PaperDetail {
  arxiv_id: string;
  title: string;
  abstract?: string;
  authors?: string[];
  published: string | null;
  pagerank: number;
  in_degree: number;
  reference_count: number;
  cited_by: number;
}

export interface Prerequisite {
  arxiv_id: string;
  title: string;
  depth: number;
  published: string | null;
}

export interface CitingContext {
  citing_id: string;
  citing_title: string | null;
  sentence: string;
  function: string;
}

export interface StudiedPaper {
  paper_id: string;
  title: string;
  times: number;
  last_at: string | null;
}
