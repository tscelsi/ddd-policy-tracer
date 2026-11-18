"""Tests for selected-chunk metadata in the terminal review workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.visualise_chunks as chunk_cli
from scripts.visualise_chunks import _row_from_payload, _save_case, parse_args


def test_chunk_viewer_retains_available_source_metadata() -> None:
    """Expose speaker, publication, and document metadata alongside passage text."""
    row = _row_from_payload(
        {
            "chunk_id": "chunk-1",
            "source_id": "hansard",
            "source_document_id": "speech-1",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 20,
            "chunk_text": "The selected passage.",
            "metadata": '{"topic":"housing"}',
            "published_at": "2025-01-10",
            "speaker_name": "Member A",
            "chamber": "senate",
        },
    )

    assert row.metadata == {
        "topic": "housing",
        "published_at": "2025-01-10",
        "speaker_name": "Member A",
        "chamber": "senate",
    }


def test_review_mode_is_opt_in() -> None:
    """Keep the original chunk viewer read-only unless review is requested."""
    assert parse_args([]).review is False
    assert parse_args(["--review", "--fixture-path", "cases.json"]).review is True
    assert str(parse_args(["--review", "--fixture-path", "cases.json"]).fixture_path) == (
        "cases.json"
    )


def test_failed_fixture_write_preserves_original_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep reviewed fixture bytes intact when staging its replacement fails."""
    fixture_path = tmp_path / "cases.json"
    original = b'{"cases":[{"case_id":"existing"}]}\n'
    fixture_path.write_bytes(original)
    fixture = {"cases": []}
    case = {
        "case_id": "candidate",
        "question": "What is the policy?",
        "answerable": True,
        "gold_evidence_chunk_ids": ["chunk-1"],
        "review_status": "draft",
    }

    def fail_sync(file_descriptor: int) -> None:
        raise OSError("temporary file sync failed")

    def fail_if_replaced(source: Path, destination: Path) -> None:
        raise AssertionError("replacement must not be attempted after a staging failure")

    monkeypatch.setattr(chunk_cli.os, "fsync", fail_sync)
    monkeypatch.setattr(chunk_cli.os, "replace", fail_if_replaced)

    with pytest.raises(OSError, match="temporary file sync failed"):
        _save_case(fixture, case, fixture_path)

    assert fixture_path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [fixture_path]
