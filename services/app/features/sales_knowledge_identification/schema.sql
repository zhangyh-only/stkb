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

CREATE TABLE IF NOT EXISTS business_entities (
    entity_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    source_mentions JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_objects (
    knowledge_object_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    domain TEXT NOT NULL,
    module TEXT NOT NULL,
    object_type TEXT NOT NULL,
    title TEXT NOT NULL,
    identity_key TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    content JSONB NOT NULL,
    entity_references JSONB NOT NULL,
    evidence JSONB NOT NULL,
    file_path TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (workspace_id, object_type, identity_key)
);

CREATE TABLE IF NOT EXISTS knowledge_object_sources (
    knowledge_object_id TEXT NOT NULL REFERENCES knowledge_objects(knowledge_object_id),
    document_package_id TEXT NOT NULL REFERENCES document_packages(document_package_id),
    candidate_id TEXT NOT NULL,
    evidence JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (knowledge_object_id, document_package_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS knowledge_build_results (
    build_id UUID PRIMARY KEY,
    run_id UUID NOT NULL UNIQUE,
    document_package_id TEXT NOT NULL REFERENCES document_packages(document_package_id),
    status TEXT NOT NULL,
    result_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_objects_module
    ON knowledge_objects(workspace_id, domain, module, object_type);
