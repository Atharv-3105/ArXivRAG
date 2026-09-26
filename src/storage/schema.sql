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
