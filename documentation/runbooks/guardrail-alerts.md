# Guardrail rejection rate high

**Symptoms:** `GuardrailRejections` fires (> 15% of answers not approved over an hour; the target is ≥ 95% approved). Users see "Check before relying on this" notices.

**Diagnose**
1. Dashboard *Guardrail issues*: which issue dominates?
2. Split by model and prompt: `python -m src.eval.ab --days 1` gives approval per `prompt_version`. The provider is on each message (`messages.provider`).
3. Pull examples: `SELECT content, checks FROM messages WHERE (checks->>'guardrail_approved')::boolean = false ORDER BY id DESC LIMIT 20;`

**Fix**

| Dominant issue | Likely cause | Action |
|---|---|---|
| `ungrounded_citations` | a model inventing `[n]` numbers | move that provider later in `LLM_TASK_ORDERS`; tighten the template's citation rule, as a new version |
| `uncited_claims` | a model ignoring "cite every claim" | same; compare template versions with the A/B report |
| `weak_support` | retrieval returning off-topic chunks | run `python -m src.eval.live retrieval`; check the corpus covers the questions being asked |
| `degraded_math_quoted` | many PDF-parsed papers | re-ingest with `--reingest` once arXiv HTML exists for them |
| One template version worse | a prompt regression | set `PROMPT_VARIANTS` back to the better version (config only, no deploy) |

**Verify:** approval returns above 95% on the dashboard.
