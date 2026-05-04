"""Script entrypoint for running one concrete claims extraction flow."""

from __future__ import annotations

import sys
from pathlib import Path

from ddd_policy_tracer.utils.events.local import LocalPublisher

from .adapters import FilesystemChunkRepository, FilesystemClaimRepository
from .extractors import RuleBasedSentenceClaimExtractor
from .models import ClaimExtractionReport
from .service_layer import ClaimsService


def run(
    *,
    chunk_id: str,
    chunk_state_path: Path,
    claim_state_path: Path,
) -> ClaimExtractionReport:
    """Run one concrete claims extraction for a single chunk identifier."""
    service = ClaimsService(
        chunk_repository=FilesystemChunkRepository(chunk_state_path),
        claim_repository=FilesystemClaimRepository(claim_state_path),
        extractor=RuleBasedSentenceClaimExtractor(),
        event_publisher=LocalPublisher(),
    )
    return service.extract_claims_for_chunk(chunk_id=chunk_id)


if __name__ == "__main__":
    report = run(
        chunk_id="chunk_395ce85ae4ce6e54",
        chunk_state_path=Path("data/chunks.jsonl"),
        claim_state_path=Path("data/claims.jsonl"),
    )
    sys.stdout.write(f"{report}\n")
