"""Unit tests for filesystem entity mention persistence behavior."""

from __future__ import annotations

from pathlib import Path

from ddd_policy_tracer.analysis.entities import EntityMention, FilesystemEntityRepository


def _sample_entity(*, entity_id: str = "entity_1") -> EntityMention:
    """Build one representative entity mention fixture record."""
    return EntityMention(
        entity_id=entity_id,
        chunk_id="chunk_123",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum="checksum-1",
        start_char=10,
        end_char=26,
        mention_text="Climate Act 2025",
        normalized_mention_text="Climate Act 2025",
        entity_type="POLICY",
        confidence=0.92,
        extractor_version="rules-v1",
    )


def test_filesystem_entity_repository_round_trips_entity_records(
    tmp_path: Path,
) -> None:
    """Persist entities to JSONL and load matching records back."""
    repository = FilesystemEntityRepository(tmp_path / "entities.jsonl")
    entity_1 = _sample_entity(entity_id="entity_1")
    entity_2 = EntityMention(
        entity_id="entity_2",
        chunk_id=entity_1.chunk_id,
        source_id=entity_1.source_id,
        source_document_id=entity_1.source_document_id,
        document_checksum=entity_1.document_checksum,
        start_char=40,
        end_char=58,
        mention_text="Australia Institute",
        normalized_mention_text="Australia Institute",
        entity_type="ORG",
        confidence=0.9,
        extractor_version=entity_1.extractor_version,
    )
    entities = [entity_1, entity_2]

    inserted = repository.add_entities(entities)

    assert inserted == 2
    assert repository.list_entities() == entities


def test_filesystem_entity_repository_filters_by_chunk_id(
    tmp_path: Path,
) -> None:
    """Filter loaded entities by chunk identity when requested."""
    repository = FilesystemEntityRepository(tmp_path / "entities.jsonl")
    entity_a = _sample_entity(entity_id="entity_a")
    entity_b = EntityMention(
        entity_id="entity_b",
        chunk_id="chunk_other",
        source_id=entity_a.source_id,
        source_document_id=entity_a.source_document_id,
        document_checksum=entity_a.document_checksum,
        start_char=entity_a.start_char,
        end_char=entity_a.end_char,
        mention_text=entity_a.mention_text,
        normalized_mention_text=entity_a.normalized_mention_text,
        entity_type=entity_a.entity_type,
        confidence=entity_a.confidence,
        extractor_version=entity_a.extractor_version,
    )
    repository.add_entities([entity_a, entity_b])

    assert repository.list_entities(chunk_id="chunk_123") == [entity_a]


def test_filesystem_entity_repository_persists_entity_records_only(
    tmp_path: Path,
) -> None:
    """Write only entity-shaped records into append-only JSONL store."""
    state_path = tmp_path / "entities.jsonl"
    repository = FilesystemEntityRepository(state_path)
    repository.add_entities([_sample_entity()])

    raw_lines = state_path.read_text(encoding="utf-8").splitlines()

    assert len(raw_lines) == 1
    assert '"entity_id"' in raw_lines[0]
    assert '"chunk_id"' in raw_lines[0]
    assert '"status"' not in raw_lines[0]
    assert '"processed_sentences"' not in raw_lines[0]


def test_filesystem_entity_repository_skips_duplicate_entity_records(
    tmp_path: Path,
) -> None:
    """Skip inserting duplicate entities using idempotency key fields."""
    repository = FilesystemEntityRepository(tmp_path / "entities.jsonl")
    entity = _sample_entity(entity_id="entity_a")

    inserted_first = repository.add_entities([entity])
    inserted_second = repository.add_entities([entity])

    assert inserted_first == 1
    assert inserted_second == 0
    assert repository.list_entities() == [entity]


def test_filesystem_entity_repository_allows_distinct_extractor_versions(
    tmp_path: Path,
) -> None:
    """Persist entities with same span when extractor version differs."""
    repository = FilesystemEntityRepository(tmp_path / "entities.jsonl")
    entity_v1 = _sample_entity(entity_id="entity_v1")
    entity_v2 = EntityMention(
        entity_id="entity_v2",
        chunk_id=entity_v1.chunk_id,
        source_id=entity_v1.source_id,
        source_document_id=entity_v1.source_document_id,
        document_checksum=entity_v1.document_checksum,
        start_char=entity_v1.start_char,
        end_char=entity_v1.end_char,
        mention_text=entity_v1.mention_text,
        normalized_mention_text=entity_v1.normalized_mention_text,
        entity_type=entity_v1.entity_type,
        confidence=entity_v1.confidence,
        extractor_version="rules-v2",
    )

    inserted = repository.add_entities([entity_v1, entity_v2])

    assert inserted == 2
    assert repository.list_entities() == [entity_v1, entity_v2]


def test_filesystem_entity_repository_allows_distinct_entity_type_same_span(
    tmp_path: Path,
) -> None:
    """Persist same-span entities when strict entity type differs."""
    repository = FilesystemEntityRepository(tmp_path / "entities.jsonl")
    entity_policy = _sample_entity(entity_id="entity_policy")
    entity_program = EntityMention(
        entity_id="entity_program",
        chunk_id=entity_policy.chunk_id,
        source_id=entity_policy.source_id,
        source_document_id=entity_policy.source_document_id,
        document_checksum=entity_policy.document_checksum,
        start_char=entity_policy.start_char,
        end_char=entity_policy.end_char,
        mention_text=entity_policy.mention_text,
        normalized_mention_text=entity_policy.normalized_mention_text,
        entity_type="PROGRAM",
        confidence=entity_policy.confidence,
        extractor_version=entity_policy.extractor_version,
    )

    inserted = repository.add_entities([entity_policy, entity_program])

    assert inserted == 2
    assert repository.list_entities() == [entity_policy, entity_program]
