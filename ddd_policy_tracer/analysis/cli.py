"""CLI entrypoint for analysis chunking and canonicalization operations."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from ddd_policy_tracer.analysis.chunks.chunking import SpacyChunker
from ddd_policy_tracer.discovery.service_layer import (
    get_source_document_versions,
)

from .canonicalization.run import main as canonicalization_main
from .chunks.chunking_models import ChunkingConfig
from .chunks.service_layer import chunk_and_persist_document_versions
from .entities.adapters import FilesystemChunkRepository
from .entities.catalog import import_seed_catalog, promote_review_items
from .entities.review_queue import SQLiteReviewQueueRepository
from .entities.retrieval import HybridCatalogRetriever


def run_cli(argv: Sequence[str], *, stdout: TextIO) -> int:
    """Execute analysis commands from the operator CLI surface."""
    parser = argparse.ArgumentParser(
        prog="ddd-policy-tracer",
        description="Run analysis operations on acquired source documents.",
    )
    subparsers = parser.add_subparsers(dest="command")

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Chunk acquired source document versions",
    )
    chunk_parser.add_argument(
        "--source",
        required=True,
        help="Source identifier to chunk (for example australia_institute).",
    )
    chunk_parser.add_argument(
        "--state-path",
        required=True,
        help="Path to acquisition state (SQLite DB or JSONL file).",
    )
    chunk_parser.add_argument(
        "--chunk-state-path",
        required=True,
        help="Path where chunk records will be persisted.",
    )
    chunk_parser.add_argument(
        "--chunk-size-chars",
        type=int,
        default=1200,
        help="Maximum characters per chunk.",
    )
    chunk_parser.add_argument(
        "--chunk-overlap-chars",
        type=int,
        default=200,
        help="Overlap characters between adjacent chunks.",
    )

    canonicalize_parser = subparsers.add_parser(
        "canonicalize",
        help="Run analysis canonicalization workflows",
    )
    canonicalize_parser.add_argument(
        "canonicalization_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to canonicalization subcommands.",
    )

    entities_catalog_parser = subparsers.add_parser(
        "entities-catalog",
        help="Manage runtime entity catalog operations",
    )
    entities_catalog_subparsers = entities_catalog_parser.add_subparsers(
        dest="entities_catalog_command",
        required=True,
    )
    entities_catalog_import_parser = entities_catalog_subparsers.add_parser(
        "import",
        help="Import curated seed rows into runtime entity catalog.",
    )
    entities_catalog_import_parser.add_argument(
        "--seed-path",
        required=True,
        help="Path to curated seed JSONL data.",
    )
    entities_catalog_import_parser.add_argument(
        "--catalog-path",
        required=True,
        help="Path to runtime SQLite catalog database.",
    )
    entities_catalog_import_parser.add_argument(
        "--vectors-path",
        required=True,
        help="Path to sidecar vectors JSON artifact.",
    )
    entities_catalog_import_parser.add_argument(
        "--catalog-version",
        default="catalog-v1",
        help="Catalog version metadata value for compatibility checks.",
    )

    entities_catalog_promote_parser = entities_catalog_subparsers.add_parser(
        "promote",
        help="Promote approved review items into runtime catalog.",
    )
    entities_catalog_promote_parser.add_argument(
        "--review-db-path",
        required=True,
        help="Path to SQLite review queue database.",
    )
    entities_catalog_promote_parser.add_argument(
        "--catalog-path",
        required=True,
        help="Path to runtime SQLite catalog database.",
    )
    entities_catalog_promote_parser.add_argument(
        "--vectors-path",
        required=True,
        help="Path to sidecar vectors JSON artifact.",
    )
    entities_catalog_promote_parser.add_argument(
        "--catalog-version",
        default="catalog-v1",
        help="Catalog version metadata value for compatibility checks.",
    )
    entities_catalog_promote_parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve obvious pending review items before promotion.",
    )
    entities_catalog_promote_parser.add_argument(
        "--auto-approve-min-score",
        type=float,
        default=0.93,
        help="Minimum decision score threshold for auto-approval.",
    )
    entities_catalog_promote_parser.add_argument(
        "--auto-approve-min-margin",
        type=float,
        default=0.15,
        help="Minimum fused-score margin between top candidates for auto-approval.",
    )

    entities_review_parser = subparsers.add_parser(
        "entities-review",
        help="Review queue operations for unresolved entity decisions.",
    )
    entities_review_subparsers = entities_review_parser.add_subparsers(
        dest="entities_review_command",
        required=True,
    )
    entities_review_list_parser = entities_review_subparsers.add_parser(
        "list",
        help="List review queue items.",
    )
    entities_review_list_parser.add_argument(
        "--review-db-path",
        required=True,
        help="Path to SQLite review queue database.",
    )
    entities_review_list_parser.add_argument(
        "--state",
        choices=["pending", "approved", "rejected", "auto_approved", "promoted"],
        default=None,
        help="Optional review state filter.",
    )

    entities_review_decide_parser = entities_review_subparsers.add_parser(
        "decide",
        help="Record explicit decision for one review item.",
    )
    entities_review_decide_parser.add_argument(
        "--review-db-path",
        required=True,
        help="Path to SQLite review queue database.",
    )
    entities_review_decide_parser.add_argument(
        "--mention-entity-id",
        required=True,
        help="Mention entity identifier in review queue.",
    )
    entities_review_decide_parser.add_argument(
        "--decision",
        required=True,
        choices=["approved", "rejected"],
        help="Review decision outcome.",
    )
    entities_review_decide_parser.add_argument(
        "--canonical-entity-key",
        default=None,
        help="Canonical key override used for approved items.",
    )
    entities_review_decide_parser.add_argument(
        "--canonical-name",
        default=None,
        help="Canonical name override used for approved items.",
    )
    entities_review_decide_parser.add_argument(
        "--entity-type",
        default=None,
        help="Entity type override used for approved items.",
    )
    entities_review_decide_parser.add_argument(
        "--aliases",
        default="",
        help="Comma-separated canonical aliases override.",
    )

    entities_review_curate_parser = entities_review_subparsers.add_parser(
        "curate",
        help="Interactively curate pending review items one at a time.",
    )
    entities_review_curate_parser.add_argument(
        "--review-db-path",
        required=True,
        help="Path to SQLite review queue database.",
    )
    entities_review_curate_parser.add_argument(
        "--chunk-state-path",
        required=True,
        help="Path to chunk JSONL state for sentence-window context.",
    )
    entities_review_curate_parser.add_argument(
        "--catalog-path",
        default=None,
        help="Optional runtime SQLite catalog path for on-demand candidate lookup.",
    )
    entities_review_curate_parser.add_argument(
        "--vectors-path",
        default=None,
        help="Optional vectors JSON path for on-demand candidate lookup.",
    )
    entities_review_curate_parser.add_argument(
        "--non-interactive-action",
        choices=["skip"],
        default=None,
        help="Optional non-interactive action for all processed items.",
    )
    entities_review_curate_parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum pending items to process in one interactive session.",
    )

    args = parser.parse_args(list(argv))
    if args.command == "canonicalize":
        forwarded = list(args.canonicalization_args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        return canonicalization_main(forwarded)

    if args.command == "entities-catalog":
        if args.entities_catalog_command == "import":
            report = import_seed_catalog(
                seed_path=Path(args.seed_path),
                catalog_path=Path(args.catalog_path),
                vectors_path=Path(args.vectors_path),
                catalog_version=args.catalog_version,
            )
            stdout.write(
                json.dumps(
                    {
                        "catalog_version": report.catalog_version,
                        "seed_hash": report.seed_hash,
                        "processed_records": report.processed_records,
                        "inserted_records": report.inserted_records,
                        "vectors_written": report.vectors_written,
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )
            return 0

        report = promote_review_items(
            review_db_path=Path(args.review_db_path),
            catalog_path=Path(args.catalog_path),
            vectors_path=Path(args.vectors_path),
            catalog_version=args.catalog_version,
            auto_approve=args.auto_approve,
            min_score=args.auto_approve_min_score,
            min_margin=args.auto_approve_min_margin,
        )
        stdout.write(
            json.dumps(
                {
                    "auto_approved_items": report.auto_approved_items,
                    "promoted_items": report.promoted_items,
                    "inserted_records": report.inserted_records,
                    "updated_records": report.updated_records,
                    "vectors_written": report.vectors_written,
                },
                ensure_ascii=True,
            )
            + "\n",
        )
        return 0

    if args.command == "entities-review":
        repository = SQLiteReviewQueueRepository(sqlite_path=Path(args.review_db_path))
        if args.entities_review_command == "list":
            items = repository.list_review_items(state=args.state)
            stdout.write(json.dumps({"items": items}, ensure_ascii=True) + "\n")
            return 0

        if args.entities_review_command == "curate":
            summary = _run_interactive_curation(
                repository=repository,
                chunk_repository=FilesystemChunkRepository(Path(args.chunk_state_path)),
                stdout=stdout,
                limit=args.limit,
                non_interactive_action=args.non_interactive_action,
                retriever=(
                    HybridCatalogRetriever(
                        catalog_path=Path(args.catalog_path),
                        vectors_path=Path(args.vectors_path),
                    )
                    if args.catalog_path is not None and args.vectors_path is not None
                    else None
                ),
            )
            stdout.write(json.dumps(summary, ensure_ascii=True) + "\n")
            return 0

        aliases = [alias.strip() for alias in str(args.aliases).split(",") if alias.strip()]
        report = repository.decide_item(
            mention_entity_id=args.mention_entity_id,
            decision=args.decision,
            canonical_entity_key=args.canonical_entity_key,
            canonical_name=args.canonical_name,
            entity_type=args.entity_type,
            aliases=aliases,
        )
        stdout.write(
            json.dumps(
                {
                    "mention_entity_id": report.mention_entity_id,
                    "review_state": report.review_state,
                    "updated_rows": report.updated_rows,
                },
                ensure_ascii=True,
            )
            + "\n",
        )
        return 0

    if args.command != "chunk":
        parser.print_help(file=stdout)
        return 2

    versions = get_source_document_versions(
        state_path=Path(args.state_path),
        source_id=args.source,
    )
    config = ChunkingConfig(
        chunk_size_chars=args.chunk_size_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
    )
    chunker = SpacyChunker(config=config)
    report = chunk_and_persist_document_versions(
        documents=versions,
        chunker=chunker,
        state_path=Path(args.chunk_state_path),
    )
    stdout.write(
        " ".join(
            [
                f"source={args.source}",
                f"processed_documents={report.processed_documents}",
                f"chunked_documents={report.chunked_documents}",
                f"skipped_documents={report.skipped_documents}",
                f"persisted_chunks={report.persisted_chunks}",
            ],
        )
        + "\n",
    )
    return 0


def _run_interactive_curation(
    *,
    repository: SQLiteReviewQueueRepository,
    chunk_repository: FilesystemChunkRepository,
    stdout: TextIO,
    limit: int,
    non_interactive_action: str | None,
    retriever: HybridCatalogRetriever | None,
) -> dict[str, int]:
    """Run one interactive review loop over pending catalog candidates."""
    pending = repository.list_pending_curation_items(limit=limit)
    processed = 0
    merged = 0
    created = 0
    skipped = 0
    for item in pending:
        processed += 1
        mention_entity_id = str(item["mention_entity_id"])
        chunk = chunk_repository.get_chunk(chunk_id=str(item["chunk_id"]))
        snippet = _build_sentence_window_snippet(
            text="" if chunk is None else chunk.chunk_text,
            start_char=int(item["start_char"]),
            end_char=int(item["end_char"]),
            mention_text=str(item["mention_text"]),
        )
        candidates = _coerce_candidate_list(item.get("top_candidates"))
        if not candidates and retriever is not None:
            retrieval_result = retriever.retrieve(
                mention_text=str(item["mention_text"]),
                normalized_mention_text=str(item["normalized_mention_text"]),
                mention_entity_type=str(item["entity_type"]),
                top_k=5,
            )
            candidates = [
                {
                    "canonical_entity_key": candidate.canonical_entity_key,
                    "canonical_name": candidate.canonical_name,
                    "entity_type": candidate.entity_type,
                    "fused_score": candidate.fused_score,
                }
                for candidate in retrieval_result.candidates
            ]

        _print_review_item(
            stdout=stdout,
            index=processed,
            total=len(pending),
            item=item,
            snippet=snippet,
            candidates=candidates,
        )
        if non_interactive_action == "skip":
            action = "s"
        else:
            action = input("Action [m=merge, n=new, s=skip, q=quit]: ").strip().casefold()
        if action == "q":
            processed -= 1
            break
        if action == "s":
            repository.decide_item(mention_entity_id=mention_entity_id, decision="rejected")
            skipped += 1
            continue
        if action == "m":
            if not candidates:
                stdout.write("No candidates available; use 'n' or 's'.\n")
                processed -= 1
                continue
            choice_raw = input(f"Candidate index [1-{len(candidates)}]: ").strip()
            if not choice_raw.isdigit():
                stdout.write("Invalid candidate index; item left pending.\n")
                processed -= 1
                continue
            choice = int(choice_raw)
            if choice < 1 or choice > len(candidates):
                stdout.write("Candidate index out of range; item left pending.\n")
                processed -= 1
                continue
            selected = candidates[choice - 1]
            aliases = [str(item["mention_text"])]
            repository.decide_item(
                mention_entity_id=mention_entity_id,
                decision="approved",
                canonical_entity_key=str(selected["canonical_entity_key"]),
                canonical_name=str(selected["canonical_name"]),
                entity_type=str(selected["entity_type"]),
                aliases=aliases,
            )
            merged += 1
            continue
        if action == "n":
            default_name = str(item["mention_text"]).strip()
            provided = input(f"Canonical name [{default_name}]: ").strip()
            canonical_name = provided or default_name
            entity_type = str(item["entity_type"]).strip().upper()
            canonical_key = _build_canonical_entity_key(
                entity_type=entity_type,
                canonical_name=canonical_name,
            )
            repository.decide_item(
                mention_entity_id=mention_entity_id,
                decision="approved",
                canonical_entity_key=canonical_key,
                canonical_name=canonical_name,
                entity_type=entity_type,
                aliases=[default_name],
            )
            created += 1
            continue

        stdout.write("Unknown action; item left pending.\n")
        processed -= 1

    return {
        "processed_items": processed,
        "merged_items": merged,
        "created_items": created,
        "skipped_items": skipped,
    }


def _coerce_candidate_list(raw: object) -> list[dict[str, object]]:
    """Normalize candidate payload values into typed candidate dictionaries."""
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = item.get("canonical_entity_key")
        name = item.get("canonical_name")
        entity_type = item.get("entity_type")
        if not isinstance(key, str) or not key.strip():
            continue
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(entity_type, str) or not entity_type.strip():
            continue
        normalized.append(
            {
                "canonical_entity_key": key.strip(),
                "canonical_name": name.strip(),
                "entity_type": entity_type.strip(),
                "fused_score": float(item.get("fused_score", 0.0)),
            },
        )
    return normalized


def _build_sentence_window_snippet(
    *,
    text: str,
    start_char: int,
    end_char: int,
    mention_text: str,
) -> str:
    """Build a sentence-window snippet around one mention span."""
    if not text:
        return ""
    bounded_start = max(0, min(start_char, len(text)))
    bounded_end = max(bounded_start, min(end_char, len(text)))
    left = max(text.rfind(".", 0, bounded_start), text.rfind("\n", 0, bounded_start))
    right_period = text.find(".", bounded_end)
    right_newline = text.find("\n", bounded_end)
    right_candidates = [candidate for candidate in [right_period, right_newline] if candidate != -1]
    right = min(right_candidates) if right_candidates else len(text)
    window_start = 0 if left == -1 else left + 1
    window_end = right if right == len(text) else right + 1
    snippet = text[window_start:window_end].strip()
    if not snippet:
        return ""
    mention_surface = text[bounded_start:bounded_end]
    if not mention_surface.strip():
        fallback_surface = text.casefold().find(mention_text.strip().casefold())
        if fallback_surface >= 0:
            mention_surface = text[fallback_surface : fallback_surface + len(mention_text.strip())]
    if not mention_surface.strip():
        return snippet
    highlighted = f"[{mention_surface}]" if mention_surface else mention_surface
    return snippet.replace(mention_surface, highlighted, 1)


def _print_review_item(
    *,
    stdout: TextIO,
    index: int,
    total: int,
    item: dict[str, object],
    snippet: str,
    candidates: list[dict[str, object]],
) -> None:
    """Print one pending review item in compact interactive format."""
    stdout.write(
        "\n".join(
            [
                f"--- Candidate {index}/{total} ---",
                f"mention_entity_id: {item['mention_entity_id']}",
                f"mention: {item['mention_text']}",
                f"entity_type: {item['entity_type']}",
                f"decision_status: {item['decision_status']} (score={item['decision_score']})",
                f"reason_codes: {','.join(str(code) for code in item['reason_codes'])}",
                f"context: {snippet}",
            ],
        )
        + "\n",
    )
    if not candidates:
        stdout.write("similar_candidates: none\n")
        return
    stdout.write("similar_candidates:\n")
    for i, candidate in enumerate(candidates, start=1):
        stdout.write(
            f"  {i}. key={candidate['canonical_entity_key']}"
            f" name={candidate['canonical_name']}"
            f" type={candidate['entity_type']}"
            f" score={candidate['fused_score']}\n",
        )


def _build_canonical_entity_key(*, entity_type: str, canonical_name: str) -> str:
    """Build deterministic canonical key from entity type and canonical name."""
    prefix_by_type = {
        "PERSON": "person",
        "ORG": "org",
        "JURISDICTION": "jurisdiction",
        "POLICY": "policy",
        "PROGRAM": "program",
    }
    prefix = prefix_by_type.get(entity_type.upper(), "entity")
    slug = re.sub(r"[^a-z0-9]+", "-", canonical_name.casefold()).strip("-")
    if not slug:
        slug = "unknown"
    return f"{prefix}:{slug}"
