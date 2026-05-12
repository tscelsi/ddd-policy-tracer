"""Unit tests for spaCy + fastcoref entity extraction behavior."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk
from ddd_policy_tracer.analysis.entities import (
    SpacyFastCorefEntityExtractor,
    SpacyFastCorefEntityExtractorConfig,
)
from ddd_policy_tracer.analysis.entities import extractors as entity_extractors_module


@dataclass(frozen=True)
class StubEntity:
    """Represent one spaCy-like entity span for extractor tests."""

    start_char: int
    end_char: int
    text: str
    label_: str


@dataclass(frozen=True)
class StubMention:
    """Represent one fastcoref-like mention span for tests."""

    start_char: int
    end_char: int
    text: str


@dataclass(frozen=True)
class StubCluster:
    """Represent one fastcoref-like cluster with ordered mentions."""

    mentions: list[StubMention]


class StubDocExtensions:
    """Expose fastcoref-like extension payload on spaCy-like doc."""

    def __init__(self, clusters: list[StubCluster]) -> None:
        """Store deterministic clusters for lookup behavior tests."""
        self.coref_clusters = clusters


@dataclass
class StubSentence:
    """Represent one sentence-like segment in spaCy-like docs."""

    text: str


class StubDoc:
    """Represent minimal spaCy-like document for extraction tests."""

    def __init__(
        self,
        *,
        ents: list[StubEntity],
        sentences: list[str],
        clusters: list[StubCluster],
    ) -> None:
        """Build deterministic doc with entities, sentences, and clusters."""
        self.ents = ents
        self.sents = [StubSentence(text=sentence) for sentence in sentences]
        self._ = StubDocExtensions(clusters)


class StubNlp:
    """Return one static spaCy-like document for any input text."""

    def __init__(self, doc: StubDoc) -> None:
        """Store deterministic document returned for parsing calls."""
        self._doc = doc

    def __call__(self, text: str) -> StubDoc:
        """Return static document while exercising input plumbing."""
        _ = text
        return self._doc


def _chunk(text: str) -> DocumentChunk:
    """Build one representative chunk fixture for extractor tests."""
    return DocumentChunk(
        chunk_id="chunk_test",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum="checksum-1",
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        chunk_text=text,
    )


def test_spacy_fastcoref_extractor_maps_entities_and_coref_key() -> None:
    """Map spaCy entities to strict types and enrich canonical entity key."""
    text = "The Clean Energy Act was discussed by the Australia Institute. It was praised."
    act_start = text.index("Clean Energy Act")
    act_end = act_start + len("Clean Energy Act")
    org_start = text.index("Australia Institute")
    org_end = org_start + len("Australia Institute")
    pronoun_start = text.index("It")
    pronoun_end = pronoun_start + len("It")

    doc = StubDoc(
        ents=[
            StubEntity(
                start_char=act_start,
                end_char=act_end,
                text="Clean Energy Act",
                label_="ORG",
            ),
            StubEntity(
                start_char=org_start,
                end_char=org_end,
                text="Australia Institute",
                label_="ORG",
            ),
        ],
        sentences=[
            "The Clean Energy Act was discussed by the Australia Institute.",
            "It was praised.",
        ],
        clusters=[
            StubCluster(
                mentions=[
                    StubMention(
                        start_char=act_start,
                        end_char=act_end,
                        text="Clean Energy Act",
                    ),
                    StubMention(
                        start_char=pronoun_start,
                        end_char=pronoun_end,
                        text="It",
                    ),
                ],
            ),
        ],
    )

    extractor = SpacyFastCorefEntityExtractor(
        config=SpacyFastCorefEntityExtractorConfig(extractor_version="spacy-fastcoref-v1"),
        nlp=StubNlp(doc),
    )

    entities = extractor.extract(chunk=_chunk(text))

    assert [entity.entity_type for entity in entities] == ["POLICY", "ORG"]
    assert entities[0].mention_text == "Clean Energy Act"
    assert entities[0].canonical_entity_key == "clean energy act"
    assert entities[0].metadata == {
        "spacy_label": "ORG",
        "coref_canonical": "Clean Energy Act",
    }


def test_spacy_fastcoref_extractor_counts_spacy_sentences() -> None:
    """Count processed sentences from spaCy-like sentence segmentation."""
    text = "One sentence. Two sentences?"
    doc = StubDoc(ents=[], sentences=["One sentence.", "Two sentences?"], clusters=[])
    extractor = SpacyFastCorefEntityExtractor(nlp=StubNlp(doc))

    processed_sentences = extractor.count_processed_sentences(chunk=_chunk(text))

    assert processed_sentences == 2


def test_spacy_fastcoref_extractor_falls_back_to_rule_when_no_spacy_entities() -> None:
    """Fallback to deterministic rule extractor when spaCy emits no entities."""
    text = "The Clean Energy Act was discussed by Australia Institute in Queensland."
    doc = StubDoc(ents=[], sentences=[text], clusters=[])
    extractor = SpacyFastCorefEntityExtractor(
        config=SpacyFastCorefEntityExtractorConfig(
            extractor_version="spacy-fastcoref-v1",
            fallback_to_rule_extractor=True,
        ),
        nlp=StubNlp(doc),
    )

    entities = extractor.extract(chunk=_chunk(text))

    assert len(entities) >= 1
    assert all(
        entity.extractor_version == "spacy-fastcoref-v1-fallback-rules"
        for entity in entities
    )


def test_build_spacy_fastcoref_nlp_logs_warning_when_model_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emit warning when configured spaCy model cannot be loaded."""
    config = SpacyFastCorefEntityExtractorConfig(spacy_model="missing-model", enable_coref=False)

    caplog.set_level(logging.WARNING)
    nlp = entity_extractors_module._build_spacy_fastcoref_nlp(config)

    assert nlp is not None
    assert "not found; falling back to blank 'en' pipeline" in caplog.text
