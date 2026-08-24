CREATE TABLE IF NOT EXISTS document_packages (
    document_package_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    source_file_name TEXT NOT NULL,
    source_file_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    full_markdown_path TEXT NOT NULL,
    full_markdown_sha256 TEXT NOT NULL,
    processing_method TEXT NOT NULL,
    status TEXT NOT NULL,
    anchors JSONB NOT NULL,
    quality_issues JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE document_packages
    ADD COLUMN IF NOT EXISTS source_file_path TEXT;

UPDATE document_packages
SET source_file_path = 'workspace/source-materials/' || document_package_id || '/' || source_file_name
WHERE source_file_path IS NULL;

ALTER TABLE document_packages
    ALTER COLUMN source_file_path SET NOT NULL;

CREATE TABLE IF NOT EXISTS sales_knowledge_identification_runs (
    run_id UUID PRIMARY KEY,
    document_package_id TEXT NOT NULL REFERENCES document_packages(document_package_id),
    status TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    result_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_identification_runs_document_package
    ON sales_knowledge_identification_runs(document_package_id, created_at DESC);
