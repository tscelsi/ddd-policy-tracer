"""CLI entrypoint for analysis canonicalization stage workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import (
    JsonlCanonicalClaimRepository,
    JsonlCanonicalEntityRepository,
    JsonlClaimSourceRepository,
    JsonlEntitySourceRepository,
)
from .models import CanonicalizationReport
from .service_layer import ClaimCanonicalizationService, EntityCanonicalizationService


def run_entities(
    *,
    entities_path: Path,
    entities_canonical_path: Path,
    entity_canonicalizer_version: str,
) -> CanonicalizationReport:
    """Run one entity canonicalization pass from explicit paths."""
    service = EntityCanonicalizationService(
        source_repository=JsonlEntitySourceRepository(path=entities_path),
        canonical_repository=JsonlCanonicalEntityRepository(path=entities_canonical_path),
        canonicalizer_version=entity_canonicalizer_version,
    )
    return service.run()


def run_claims(
    *,
    claims_path: Path,
    entities_canonical_path: Path,
    claims_canonical_path: Path,
    claim_canonicalizer_version: str,
    required_entity_canonicalizer_version: str,
) -> CanonicalizationReport:
    """Run one claim canonicalization pass from explicit paths."""
    service = ClaimCanonicalizationService(
        claim_source_repository=JsonlClaimSourceRepository(path=claims_path),
        canonical_entity_repository=JsonlCanonicalEntityRepository(path=entities_canonical_path),
        canonical_claim_repository=JsonlCanonicalClaimRepository(path=claims_canonical_path),
        claim_canonicalizer_version=claim_canonicalizer_version,
        required_entity_canonicalizer_version=required_entity_canonicalizer_version,
    )
    return service.run()


def run_all(
    *,
    entities_path: Path,
    entities_canonical_path: Path,
    claims_path: Path,
    claims_canonical_path: Path,
    entity_canonicalizer_version: str,
    claim_canonicalizer_version: str,
) -> tuple[CanonicalizationReport, CanonicalizationReport]:
    """Run entity then claim canonicalization in strict dependency order."""
    entity_report = run_entities(
        entities_path=entities_path,
        entities_canonical_path=entities_canonical_path,
        entity_canonicalizer_version=entity_canonicalizer_version,
    )
    claim_report = run_claims(
        claims_path=claims_path,
        entities_canonical_path=entities_canonical_path,
        claims_canonical_path=claims_canonical_path,
        claim_canonicalizer_version=claim_canonicalizer_version,
        required_entity_canonicalizer_version=entity_canonicalizer_version,
    )
    return entity_report, claim_report


def _build_parser() -> argparse.ArgumentParser:
    """Build parser for canonicalization subcommands."""
    parser = argparse.ArgumentParser(
        prog="analysis-canonicalization",
        description="Run entity and claim canonicalization workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    entity_parser = subparsers.add_parser(
        "entities",
        help="Canonicalize entity extraction outputs.",
    )
    entity_parser.add_argument("--entities-path", default="data/entities.jsonl")
    entity_parser.add_argument(
        "--entities-canonical-path",
        default="data/entities_canonical.jsonl",
    )
    entity_parser.add_argument(
        "--entity-canonicalizer-version",
        default="entity-v1",
    )

    claims_parser = subparsers.add_parser(
        "claims",
        help="Canonicalize claim extraction outputs.",
    )
    claims_parser.add_argument("--claims-path", default="data/claims.jsonl")
    claims_parser.add_argument(
        "--entities-canonical-path",
        default="data/entities_canonical.jsonl",
    )
    claims_parser.add_argument(
        "--claims-canonical-path",
        default="data/claims_canonical.jsonl",
    )
    claims_parser.add_argument(
        "--claim-canonicalizer-version",
        default="claim-v1",
    )
    claims_parser.add_argument(
        "--required-entity-canonicalizer-version",
        default="entity-v1",
    )

    all_parser = subparsers.add_parser(
        "all",
        help="Run entity and claim canonicalization in dependency order.",
    )
    all_parser.add_argument("--entities-path", default="data/entities.jsonl")
    all_parser.add_argument(
        "--entities-canonical-path",
        default="data/entities_canonical.jsonl",
    )
    all_parser.add_argument("--claims-path", default="data/claims.jsonl")
    all_parser.add_argument(
        "--claims-canonical-path",
        default="data/claims_canonical.jsonl",
    )
    all_parser.add_argument(
        "--entity-canonicalizer-version",
        default="entity-v1",
    )
    all_parser.add_argument(
        "--claim-canonicalizer-version",
        default="claim-v1",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run canonicalization command from command-line arguments."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "entities":
        report = run_entities(
            entities_path=Path(args.entities_path),
            entities_canonical_path=Path(args.entities_canonical_path),
            entity_canonicalizer_version=args.entity_canonicalizer_version,
        )
        sys.stdout.write(json.dumps(report.__dict__, ensure_ascii=True) + "\n")
        return 0

    if args.command == "claims":
        report = run_claims(
            claims_path=Path(args.claims_path),
            entities_canonical_path=Path(args.entities_canonical_path),
            claims_canonical_path=Path(args.claims_canonical_path),
            claim_canonicalizer_version=args.claim_canonicalizer_version,
            required_entity_canonicalizer_version=args.required_entity_canonicalizer_version,
        )
        sys.stdout.write(json.dumps(report.__dict__, ensure_ascii=True) + "\n")
        return 0

    entity_report, claim_report = run_all(
        entities_path=Path(args.entities_path),
        entities_canonical_path=Path(args.entities_canonical_path),
        claims_path=Path(args.claims_path),
        claims_canonical_path=Path(args.claims_canonical_path),
        entity_canonicalizer_version=args.entity_canonicalizer_version,
        claim_canonicalizer_version=args.claim_canonicalizer_version,
    )
    sys.stdout.write(
        json.dumps(
            {
                "entity": entity_report.__dict__,
                "claim": claim_report.__dict__,
            },
            ensure_ascii=True,
        )
        + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
