-- Applied by src/storage/database.py:init_db_schema (the single source of truth for the schema).
-- Statements are split on ';' by the loader, so do not put ';' inside strings or comments.
-- Every statement is idempotent: re-running it upgrades an existing database in place.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS papers(
    arxiv_id        VARCHAR(30) PRIMARY KEY,
    title           TEXT        NOT NULL,
    abstract        TEXT        NOT NULL,
    authors         JSONB       NOT NULL,
    published_date  TIMESTAMPTZ NOT NULL,
    pdf_url         TEXT,
    categories      JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- How the paper was parsed. math_fidelity is 'latex' (real LaTeX from arXiv HTML) or 'degraded'
-- (recovered from PDF glyphs), so downstream agents know how far to trust an equation.
ALTER TABLE papers ADD COLUMN IF NOT EXISTS parse_method   VARCHAR(16);
ALTER TABLE papers ADD COLUMN IF NOT EXISTS math_fidelity  VARCHAR(16);
ALTER TABLE papers ADD COLUMN IF NOT EXISTS parse_stats    JSONB;

-- Consistency between Postgres (source of truth) and Qdrant (derived index): a paper is written here
-- as 'pending', and flipped to 'indexed' only after its vectors are durably in Qdrant. A crash in
-- between leaves 'pending', which the next run repairs by re-embedding the stored chunks.
-- Pre-existing rows were written by the old pipeline after a successful upsert, hence the default.
ALTER TABLE papers ADD COLUMN IF NOT EXISTS index_status  VARCHAR(16) NOT NULL DEFAULT 'indexed';

-- arXiv id without its version ("2401.12345v2" -> "2401.12345"): references cite papers without
-- versions, so citation edges join on this.
ALTER TABLE papers ADD COLUMN IF NOT EXISTS base_id VARCHAR(30)
    GENERATED ALWAYS AS (regexp_replace(arxiv_id, 'v[0-9]+$', '')) STORED;

CREATE TABLE IF NOT EXISTS paper_chunks(
    chunk_id       VARCHAR(100) PRIMARY KEY,
    paper_id       VARCHAR(30)  NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    section_title  VARCHAR(255) NOT NULL,
    content        TEXT         NOT NULL,
    metadata       JSONB,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Display equations as real LaTeX, addressable on their own (inline math stays inside chunk text).
CREATE TABLE IF NOT EXISTS paper_equations(
    paper_id      VARCHAR(30)  NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    ordinal       INT          NOT NULL,
    latex         TEXT         NOT NULL,
    number        VARCHAR(32),
    display       BOOLEAN      NOT NULL DEFAULT TRUE,
    section_path  TEXT,
    anchor        VARCHAR(100),
    PRIMARY KEY (paper_id, ordinal)
);

CREATE TABLE IF NOT EXISTS paper_references(
    paper_id  VARCHAR(30)  NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    ordinal   INT          NOT NULL,
    ref_key   VARCHAR(100),
    text      TEXT         NOT NULL,
    urls      JSONB,
    PRIMARY KEY (paper_id, ordinal)
);

-- Citation graph (Postgres instead of Neo4j - see documentation/adr/0003). One row per reference
-- that names an arXiv paper. The cited paper may or may not be in our corpus.
CREATE TABLE IF NOT EXISTS paper_citations(
    citing_id      VARCHAR(30) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    cited_base_id  VARCHAR(30) NOT NULL,
    ref_ordinal    INT,
    PRIMARY KEY (citing_id, cited_base_id)
);

-- One row per pipeline run: an audit trail and the source for parse-quality metrics.
CREATE TABLE IF NOT EXISTS ingestion_runs(
    id           BIGSERIAL PRIMARY KEY,
    query        TEXT,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at  TIMESTAMPTZ,
    status       VARCHAR(16) NOT NULL DEFAULT 'running',
    metrics      JSONB
);

-- Dead-letter queue. Deliberately NO foreign key to papers: a paper that failed
-- before it was ever stored must still be recordable.
CREATE TABLE IF NOT EXISTS failed_papers(
    arxiv_id        VARCHAR(30) PRIMARY KEY,
    failure_reason  TEXT        NOT NULL,
    attempted_at    TIMESTAMPTZ DEFAULT NOW(),
    retry_count     INT         DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_papers_published  ON papers(published_date DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_paper_id   ON paper_chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section    ON paper_chunks(section_title);
CREATE INDEX IF NOT EXISTS idx_papers_categories ON papers USING GIN (categories);
CREATE INDEX IF NOT EXISTS idx_papers_title      ON papers USING GIN (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_papers_base_id    ON papers(base_id);
CREATE INDEX IF NOT EXISTS idx_papers_index_status ON papers(index_status) WHERE index_status <> 'indexed';
CREATE INDEX IF NOT EXISTS idx_citations_cited   ON paper_citations(cited_base_id);

-- Chat history (ADR 0007). Owned by us, not by LangGraph's checkpointer: it is queryable (history
-- UI, GDPR export/delete, eval sampling) and survives restarts of a free-tier host.
CREATE TABLE IF NOT EXISTS conversations(
    id          UUID PRIMARY KEY,
    title       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages(
    id               BIGSERIAL PRIMARY KEY,
    conversation_id  UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             VARCHAR(16) NOT NULL,
    content          TEXT NOT NULL,
    intent           VARCHAR(16),
    citations        JSONB,
    provider         VARCHAR(120),
    trace_id         VARCHAR(128),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);

-- ---------------------------------------------------------------- Phase 3: users & memory (ADR 0008)

CREATE TABLE IF NOT EXISTS users(
    id             BIGSERIAL PRIMARY KEY,
    email          VARCHAR(320) NOT NULL UNIQUE,          -- stored lower-cased
    password_hash  TEXT NOT NULL,                         -- bcrypt
    display_name   VARCHAR(100),
    is_admin       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Conversations belong to a user. Rows from before auth existed keep NULL and are unreachable.
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id) ON DELETE CASCADE;
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC);

-- One row per user. `profile` holds every field, `locked` the ones the user set explicitly: inference
-- never overwrites a locked field. `version` gives optimistic concurrency for the JSONB patches.
CREATE TABLE IF NOT EXISTS user_profiles(
    user_id     BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    profile     JSONB NOT NULL DEFAULT '{}'::jsonb,
    locked      JSONB NOT NULL DEFAULT '[]'::jsonb,
    version     INT NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Per-concept mastery with spaced-repetition schedule. `mastery` is the value at `updated_at`;
-- decay toward the prior is applied when read, so no background job has to rewrite rows daily.
CREATE TABLE IF NOT EXISTS concept_mastery(
    user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    concept         VARCHAR(80) NOT NULL,
    mastery         REAL NOT NULL CHECK (mastery >= 0 AND mastery <= 1),
    evidence        INT NOT NULL DEFAULT 0,
    locked          BOOLEAN NOT NULL DEFAULT FALSE,
    ease            REAL NOT NULL DEFAULT 2.5,
    interval_days   REAL NOT NULL DEFAULT 0,
    reviews         INT NOT NULL DEFAULT 0,
    next_review_at  TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, concept)
);
CREATE INDEX IF NOT EXISTS idx_mastery_due ON concept_mastery(user_id, next_review_at);

-- Every processed signal. The primary key is the event's idempotency key: a redelivered event is a
-- no-op because the insert conflicts inside the same transaction that applies its effects.
CREATE TABLE IF NOT EXISTS user_events(
    event_id    UUID PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind        VARCHAR(32) NOT NULL,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_events_user ON user_events(user_id, created_at DESC);

-- Transparent audit trail: every change to a profile field or a concept's mastery, and why.
CREATE TABLE IF NOT EXISTS profile_audit(
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    field       VARCHAR(120) NOT NULL,                    -- "expertise_level" or "mastery:lora"
    old_value   JSONB,
    new_value   JSONB,
    source      VARCHAR(16) NOT NULL,                     -- explicit | inferred
    event_id    UUID,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_user ON profile_audit(user_id, created_at DESC);

-- ---------------------------------------------------------------- Phase 4: graph (ADR 0009)

-- The sentence in paper A that cites arXiv paper B, and what it does with B (critical,
-- comparative, builds_on, neutral). HTML-parsed papers only. Enables "who criticises B".
CREATE TABLE IF NOT EXISTS citation_contexts(
    id             BIGSERIAL PRIMARY KEY,
    citing_id      VARCHAR(30) NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    cited_base_id  VARCHAR(30) NOT NULL,
    ref_ordinal    INT,
    sentence       TEXT NOT NULL,
    anchor         VARCHAR(100),
    function       VARCHAR(16) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_contexts_cited  ON citation_contexts(cited_base_id, function);
CREATE INDEX IF NOT EXISTS idx_contexts_citing ON citation_contexts(citing_id);

-- Influence over the in-corpus citation graph, recomputed after ingestion (not per request).
CREATE TABLE IF NOT EXISTS paper_influence(
    arxiv_id     VARCHAR(30) PRIMARY KEY REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    in_degree    INT NOT NULL,
    pagerank     REAL NOT NULL,
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_influence_rank ON paper_influence(pagerank DESC);
CREATE INDEX IF NOT EXISTS idx_citations_citing ON paper_citations(citing_id);

-- ---------------------------------------------------------------- Phase 5: agents (ADR 0010)

-- Per-conversation agent state (the tutor session: pending question, hints, reading position).
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS state JSONB;
-- Which specialist answered, with which prompt template version (A/B), and its output checks.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS agent VARCHAR(16);
ALTER TABLE messages ADD COLUMN IF NOT EXISTS prompt_version VARCHAR(60);
ALTER TABLE messages ADD COLUMN IF NOT EXISTS checks JSONB;
-- How the answer was made (route, retrieval, timings, verdict, guardrail): what "How this was
-- answered" shows after a reload.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS trace JSONB;
CREATE INDEX IF NOT EXISTS idx_messages_prompt ON messages(prompt_version) WHERE prompt_version IS NOT NULL;
