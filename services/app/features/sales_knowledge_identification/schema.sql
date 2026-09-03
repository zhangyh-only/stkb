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

ALTER TABLE knowledge_object_sources
    ADD COLUMN IF NOT EXISTS source_occurrence_id BIGSERIAL,
    ADD COLUMN IF NOT EXISTS run_id UUID,
    ADD COLUMN IF NOT EXISTS source_claim_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS claim_usage JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint constraint_record
        JOIN unnest(constraint_record.conkey) AS key(attnum) ON TRUE
        JOIN pg_attribute attribute_record
          ON attribute_record.attrelid = constraint_record.conrelid
         AND attribute_record.attnum = key.attnum
        WHERE constraint_record.conrelid = 'knowledge_object_sources'::regclass
          AND constraint_record.contype = 'p'
          AND attribute_record.attname <> 'source_occurrence_id'
    ) THEN
        ALTER TABLE knowledge_object_sources
            DROP CONSTRAINT knowledge_object_sources_pkey;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'knowledge_object_sources'::regclass AND contype = 'p'
    ) THEN
        ALTER TABLE knowledge_object_sources
            ADD CONSTRAINT knowledge_object_sources_pkey
            PRIMARY KEY (source_occurrence_id);
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_object_source_run
    ON knowledge_object_sources (
        knowledge_object_id, document_package_id, run_id, candidate_id
    )
    WHERE run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_knowledge_object_sources_active_package
    ON knowledge_object_sources (document_package_id, active);

CREATE TABLE IF NOT EXISTS knowledge_object_lineages (
    workspace_id TEXT NOT NULL,
    source_lineage_key TEXT NOT NULL,
    knowledge_object_id TEXT NOT NULL REFERENCES knowledge_objects(knowledge_object_id),
    document_package_id TEXT NOT NULL REFERENCES document_packages(document_package_id),
    module TEXT NOT NULL,
    object_type TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (workspace_id, source_lineage_key)
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

CREATE TABLE IF NOT EXISTS knowledge_relationships (
    relationship_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_revision INTEGER,
    target_ref TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_revision INTEGER,
    direction TEXT NOT NULL DEFAULT 'forward',
    inverse_label TEXT NOT NULL,
    scope JSONB NOT NULL DEFAULT '{}'::jsonb,
    effective_period JSONB NOT NULL DEFAULT '{}'::jsonb,
    evidence JSONB NOT NULL,
    provenance JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE knowledge_relationships
    ADD COLUMN IF NOT EXISTS source_revision INTEGER,
    ADD COLUMN IF NOT EXISTS target_revision INTEGER,
    ADD COLUMN IF NOT EXISTS direction TEXT NOT NULL DEFAULT 'forward',
    ADD COLUMN IF NOT EXISTS inverse_label TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS scope JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS effective_period JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS provenance JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_knowledge_relationships_source
    ON knowledge_relationships(workspace_id, source_ref, relation_type);

CREATE INDEX IF NOT EXISTS idx_knowledge_relationships_target
    ON knowledge_relationships(workspace_id, target_ref, relation_type);

CREATE TABLE IF NOT EXISTS knowledge_retrieval_units (
    retrieval_unit_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    document_package_id TEXT NOT NULL REFERENCES document_packages(document_package_id),
    knowledge_object_id TEXT NOT NULL REFERENCES knowledge_objects(knowledge_object_id),
    revision INTEGER NOT NULL,
    domain TEXT NOT NULL,
    module TEXT NOT NULL,
    object_type TEXT NOT NULL,
    title TEXT NOT NULL,
    retrieval_text TEXT NOT NULL,
    source_file_sha256 TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    embedding vector(1024) NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_retrieval_units_active
    ON knowledge_retrieval_units(workspace_id, document_package_id, active);

CREATE INDEX IF NOT EXISTS idx_knowledge_retrieval_units_embedding
    ON knowledge_retrieval_units USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS knowledge_relationship_sources (
    relationship_id TEXT NOT NULL REFERENCES knowledge_relationships(relationship_id),
    document_package_id TEXT NOT NULL REFERENCES document_packages(document_package_id),
    run_id UUID NOT NULL,
    evidence JSONB NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (relationship_id, document_package_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_relationship_sources_active_package
    ON knowledge_relationship_sources(document_package_id, active);
