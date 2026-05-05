"""Build claims silver datasets from chunks using LLM labels."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from ddd_policy_tracer.analysis.silver_dataset import validate_claim_silver_record
from ddd_policy_tracer.utils.logger import configure_logging, get_logger

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
LOGGER = get_logger(__name__, ctx="claims_silver_dataset")


class HttpClient(Protocol):
    """Minimal HTTP client protocol for LLM dataset generation."""

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> httpx.Response:
        """Send one JSON request and return response payload."""


def run(
    *,
    chunks_path: Path,
    output_path: Path,
    sample_size: int,
    seed: int,
    model: str,
    max_claims_per_chunk: int,
    sleep_seconds: float,
    label_prompt_version: str,
    dataset_version: str,
    labeling_run_id: str | None,
    labeled_at_utc: str | None,
    summary_output_path: Path | None,
    http_client: HttpClient | None = None,
) -> dict[str, object]:
    """Create claims silver JSONL with one record per claim span."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY must be set for LLM silver dataset generation")
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")

    records = _load_chunks(chunks_path)
    if not records:
        raise ValueError("No chunks found in chunks_path")

    run_id = labeling_run_id or _build_run_id(seed=seed, model=model)
    labeled_timestamp = labeled_at_utc or _now_utc_iso()
    sampled_records = _sample_chunks(records=records, sample_size=sample_size, seed=seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    close_client = False
    client: HttpClient
    if http_client is not None:
        client = http_client
    else:
        close_client = True
        client = httpx.Client(timeout=60.0)

    diagnostics: dict[str, object] = {
        "labeling_run_id": run_id,
        "dataset_version": dataset_version,
        "sampled_chunks": len(sampled_records),
        "records_written": 0,
        "claims_written": 0,
        "parse_failures": 0,
        "invalid_rows": 0,
        "chunk_failures": [],
    }

    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for index, chunk in enumerate(sampled_records, start=1):
                parse_error: str | None = None
                try:
                    claims = _extract_claims_with_llm(
                        client=client,
                        api_key=api_key,
                        model=model,
                        chunk_text=chunk["chunk_text"],
                        max_claims_per_chunk=max_claims_per_chunk,
                    )
                except (httpx.HTTPError, ValueError) as exc:
                    claims = []
                    parse_error = str(exc)

                deduped_claims = _dedupe_claims(claims)
                silver_claims: list[dict[str, int]] = []
                for claim in deduped_claims:
                    span = _claim_span(chunk_text=chunk["chunk_text"], claim_text=claim)
                    if span is not None:
                        silver_claims.append(span)

                silver_records = _to_claim_rows(
                    chunk=chunk,
                    run_id=run_id,
                    model=model,
                    label_prompt_version=label_prompt_version,
                    dataset_version=dataset_version,
                    labeled_timestamp=labeled_timestamp,
                    silver_claims=silver_claims,
                )

                try:
                    for silver_record in silver_records:
                        validate_claim_silver_record(silver_record)
                except ValueError as exc:
                    diagnostics["invalid_rows"] = int(diagnostics["invalid_rows"]) + 1
                    _append_chunk_failure(
                        diagnostics=diagnostics,
                        chunk_id=chunk["chunk_id"],
                        error_type="schema_validation_failed",
                        message=str(exc),
                    )
                    continue

                if parse_error is not None:
                    diagnostics["parse_failures"] = int(diagnostics["parse_failures"]) + 1
                    _append_chunk_failure(
                        diagnostics=diagnostics,
                        chunk_id=chunk["chunk_id"],
                        error_type="llm_parse_failed",
                        message=parse_error,
                    )

                for silver_record in silver_records:
                    handle.write(json.dumps(silver_record, ensure_ascii=True) + "\n")
                diagnostics["records_written"] = int(diagnostics["records_written"]) + len(
                    silver_records,
                )
                diagnostics["claims_written"] = int(diagnostics["claims_written"]) + len(
                    silver_claims,
                )

                sys.stdout.write(
                    f"[{index}/{len(sampled_records)}] "
                    f"chunk_id={chunk['chunk_id']} "
                    f"silver_claim_rows={len(silver_records)} "
                    f"silver_claims={len(silver_claims)}\n",
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
    finally:
        if close_client:
            cast_client = client
            if isinstance(cast_client, httpx.Client):
                cast_client.close()

    diagnostics["output_path"] = str(output_path)
    if summary_output_path is not None:
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_path.write_text(
            json.dumps(diagnostics, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        diagnostics["summary_output_path"] = str(summary_output_path)

    LOGGER.bind(
        sampled_chunks=diagnostics["sampled_chunks"],
        records_written=diagnostics["records_written"],
        parse_failures=diagnostics["parse_failures"],
        invalid_rows=diagnostics["invalid_rows"],
    ).info("claims silver dataset generation completed")
    return diagnostics


def _append_chunk_failure(
    *,
    diagnostics: dict[str, object],
    chunk_id: str,
    error_type: str,
    message: str,
) -> None:
    """Append one deterministic chunk failure record to diagnostics."""
    failures = diagnostics["chunk_failures"]
    if not isinstance(failures, list):
        raise ValueError("chunk_failures diagnostics container must be a list")
    failures.append(
        {
            "chunk_id": chunk_id,
            "error_type": error_type,
            "message": message,
        },
    )


def _sample_chunks(
    *,
    records: list[dict[str, str]],
    sample_size: int,
    seed: int,
) -> list[dict[str, str]]:
    """Sample chunk records deterministically from loaded source chunks."""
    active_sample_size = min(sample_size, len(records))
    rng = random.Random(seed)  # noqa: S311 - deterministic non-crypto sampling
    return rng.sample(records, k=active_sample_size)


def _load_chunks(chunks_path: Path) -> list[dict[str, str]]:
    """Load chunk records from JSONL and validate required chunk fields."""
    if not chunks_path.exists():
        raise ValueError("chunks_path does not exist")

    records: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(
        chunks_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        chunk = _validate_chunk_payload(payload=payload, line_number=line_number)
        records.append(chunk)
    return records


def _validate_chunk_payload(
    *,
    payload: object,
    line_number: int,
) -> dict[str, str]:
    """Validate one chunk input row contains required string fields."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"Chunk row {line_number} must be an object")

    chunk: dict[str, str] = {}
    for field_name in (
        "chunk_id",
        "source_id",
        "source_document_id",
        "document_checksum",
        "chunk_text",
    ):
        raw_value = payload.get(field_name)
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"Chunk row {line_number} has invalid {field_name}")
        chunk[field_name] = raw_value
    return chunk


def _extract_claims_with_llm(
    *,
    client: HttpClient,
    api_key: str,
    model: str,
    chunk_text: str,
    max_claims_per_chunk: int,
) -> list[str]:
    """Call LLM and parse claims list from the response payload."""
    prompt = (
        "Extract policy-relevant factual and normative claims from the text. "
        "Return only claims that are exact substrings from input text. "
        "Return JSON object with key claims where claims is an array of strings. "
        f"Limit to at most {max_claims_per_chunk} claims."
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict policy claim extraction assistant.",
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nTEXT:\n{chunk_text}",
            },
        ],
        "response_format": {"type": "json_object"},
    }
    response = client.post(
        OPENAI_CHAT_COMPLETIONS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    response.raise_for_status()
    parsed = response.json()
    try:
        content = parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("LLM response missing choices.message.content") from exc

    try:
        parsed_content = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response content is not valid JSON") from exc

    claims = parsed_content.get("claims")
    if not isinstance(claims, list):
        raise ValueError("LLM response JSON must include array field 'claims'")

    values: list[str] = []
    for raw_value in claims:
        as_string = str(raw_value).strip()
        if as_string:
            values.append(as_string)
    return values


def _dedupe_claims(claims: list[str]) -> list[str]:
    """Dedupe claims by normalized text while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        normalized = _normalize_text(claim)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(claim)
    return deduped


def _claim_span(*, chunk_text: str, claim_text: str) -> dict[str, int] | None:
    """Resolve one claim string to chunk-local start/end offsets."""
    start_char = chunk_text.find(claim_text)
    if start_char == -1:
        normalized_claim = _normalize_text(claim_text)
        if not normalized_claim:
            return None
        start_char = chunk_text.find(normalized_claim)
        if start_char == -1:
            return None
        claim_text = normalized_claim

    end_char = start_char + len(claim_text)
    return {
        "start_char": start_char,
        "end_char": end_char,
    }


def _to_claim_rows(
    *,
    chunk: dict[str, str],
    run_id: str,
    model: str,
    label_prompt_version: str,
    dataset_version: str,
    labeled_timestamp: str,
    silver_claims: list[dict[str, int]],
) -> list[dict[str, object]]:
    """Convert one chunk and its claims into claim-per-row silver records."""
    common_fields: dict[str, object] = {
        "chunk_id": chunk["chunk_id"],
        "source_id": chunk["source_id"],
        "source_document_id": chunk["source_document_id"],
        "document_checksum": chunk["document_checksum"],
        "chunk_text": chunk["chunk_text"],
        "labeling_run_id": run_id,
        "labeler_kind": "llm",
        "labeler_version": model,
        "label_prompt_version": label_prompt_version,
        "dataset_version": dataset_version,
        "labeled_at_utc": labeled_timestamp,
    }
    if not silver_claims:
        return [{**common_fields, "silver_claims": []}]
    return [{**common_fields, "silver_claims": [claim]} for claim in silver_claims]


def _normalize_text(value: str) -> str:
    """Normalize text whitespace for deterministic persistence."""
    return " ".join(value.split())


def _build_run_id(*, seed: int, model: str) -> str:
    """Build one traceable default run identifier string."""
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"claims_silver_{timestamp}_{model}_seed{seed}"


def _now_utc_iso() -> str:
    """Return current UTC timestamp in canonical ISO-8601 format."""
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for claims silver dataset generation."""
    parser = argparse.ArgumentParser(
        prog="claims-eval-build-silver-dataset",
        description="Build claims silver dataset JSONL using LLM labels.",
    )
    parser.add_argument("--chunks-path", required=True, help="Path to chunk JSONL input.")
    parser.add_argument("--output-path", required=True, help="Path to write silver JSONL output.")
    parser.add_argument(
        "--summary-output-path",
        default=None,
        help="Optional path to write diagnostics summary JSON.",
    )
    parser.add_argument("--sample-size", type=int, default=300, help="Number of chunks to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Chat completion model id.")
    parser.add_argument(
        "--max-claims-per-chunk",
        type=int,
        default=8,
        help="Maximum claims requested from labeler per chunk.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between requests.",
    )
    parser.add_argument(
        "--label-prompt-version",
        default="claims-prompt-v1",
        help="Version id for the labeling prompt.",
    )
    parser.add_argument(
        "--dataset-version",
        default="claims-silver-v1",
        help="Version id for the generated silver dataset.",
    )
    parser.add_argument(
        "--labeling-run-id",
        default=None,
        help="Optional stable run id override for lineage.",
    )
    parser.add_argument(
        "--labeled-at-utc",
        default=None,
        help="Optional UTC timestamp override for lineage.",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="warning",
        help="Log verbosity for run diagnostics.",
    )
    return parser


def main() -> int:
    """Run claims silver dataset generation from CLI arguments."""
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    summary = run(
        chunks_path=Path(args.chunks_path),
        output_path=Path(args.output_path),
        sample_size=args.sample_size,
        seed=args.seed,
        model=args.model,
        max_claims_per_chunk=args.max_claims_per_chunk,
        sleep_seconds=args.sleep_seconds,
        label_prompt_version=args.label_prompt_version,
        dataset_version=args.dataset_version,
        labeling_run_id=args.labeling_run_id,
        labeled_at_utc=args.labeled_at_utc,
        summary_output_path=(Path(args.summary_output_path) if args.summary_output_path else None),
    )
    sys.stdout.write(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
