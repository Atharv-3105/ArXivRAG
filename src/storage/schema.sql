-- Applied by src/storage/database.py:init_db_schema (the single source of truth for the schema).
-- Statements are split on ';' by the loader, so do not put ';' inside strings or comments.

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

CREATE TABLE IF NOT EXISTS paper_chunks(
    chunk_id       VARCHAR(100) PRIMARY KEY,
    paper_id       VARCHAR(30)  NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    section_title  VARCHAR(255) NOT NULL,
    content        TEXT         NOT NULL,
    metadata       JSONB,
    created_at     TIMESTAMPTZ DEFAULT NOW()
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
