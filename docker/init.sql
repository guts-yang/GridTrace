-- =====================================================================
-- GridTrace · PostgreSQL + pgvector schema
-- Run via: psql -U postgres -d gridtrace -f docker/init.sql
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------
-- kb_anchors: grid-quantized anchor points (L1 routing table)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kb_anchors (
    id            BIGSERIAL PRIMARY KEY,
    quant_key     TEXT        NOT NULL UNIQUE,
    anchor_vec    vector(512) NOT NULL,        -- quantized vector
    ref_count     INTEGER     NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_kb_anchors_quant_key ON kb_anchors (quant_key);

-- ---------------------------------------------------------------------
-- kb_entries: full-precision joint vectors (L2 rerank pool)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kb_entries (
    id            BIGSERIAL PRIMARY KEY,
    anchor_id     BIGINT      NOT NULL REFERENCES kb_anchors(id) ON DELETE CASCADE,
    question      TEXT        NOT NULL,
    solution      TEXT        NOT NULL,
    category      TEXT,
    doc_id        TEXT        NOT NULL,
    page_index    INTEGER     NOT NULL DEFAULT 0,
    match_score   REAL        NOT NULL DEFAULT 0.0,
    embedding     vector(512) NOT NULL,        -- full-precision joint vector
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, page_index, question)
);

CREATE INDEX IF NOT EXISTS idx_kb_entries_anchor_id  ON kb_entries (anchor_id);
CREATE INDEX IF NOT EXISTS idx_kb_entries_doc_id     ON kb_entries (doc_id);
CREATE INDEX IF NOT EXISTS idx_kb_entries_category   ON kb_entries (category);

-- ---------------------------------------------------------------------
-- Audit log for unlearning operations
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kb_unlearn_log (
    id            BIGSERIAL PRIMARY KEY,
    doc_id        TEXT        NOT NULL,
    entries_deleted INTEGER   NOT NULL,
    anchors_pruned  INTEGER   NOT NULL,
    triggered_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_kb_unlearn_log_doc_id ON kb_unlearn_log (doc_id);

-- ---------------------------------------------------------------------
-- Health check view
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_kb_stats AS
SELECT
    (SELECT COUNT(*) FROM kb_anchors)                       AS total_anchors,
    (SELECT COUNT(*) FROM kb_entries)                       AS total_entries,
    (SELECT COUNT(DISTINCT doc_id) FROM kb_entries)         AS unique_docs,
    (SELECT AVG(match_score) FROM kb_entries)               AS avg_match_score,
    (SELECT COUNT(*) FROM kb_anchors WHERE ref_count = 0)   AS orphan_anchors;

-- ---------------------------------------------------------------------
-- Auto-maintain ref_count on kb_anchors
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION fn_kb_anchors_bump() RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE kb_anchors SET ref_count = ref_count + 1 WHERE id = NEW.anchor_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE kb_anchors SET ref_count = ref_count - 1 WHERE id = OLD.anchor_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_kb_anchors_refcount ON kb_entries;
CREATE TRIGGER trg_kb_anchors_refcount
AFTER INSERT OR DELETE ON kb_entries
FOR EACH ROW EXECUTE FUNCTION fn_kb_anchors_bump();

-- ---------------------------------------------------------------------
-- Auto-update updated_at on kb_entries
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION fn_kb_entries_touch() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_kb_entries_touch ON kb_entries;
CREATE TRIGGER trg_kb_entries_touch
BEFORE UPDATE ON kb_entries
FOR EACH ROW EXECUTE FUNCTION fn_kb_entries_touch();
