"""Build a claim-validation dataset from chunks using an LLM annotator."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import httpx

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


def run(
    *,
    chunks_path: Path,
    output_path: Path,
    sample_size: int,
    seed: int,
    model: str,
    max_claims_per_chunk: int,
    sleep_seconds: float,
) -> None:
    """Create labeled claim records for a sampled chunk subset."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY must be set for LLM dataset generation")
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")

    chunks = _load_chunks(chunks_path)
    if not chunks:
        raise ValueError("No chunks found in chunks_path")

    active_sample_size = min(sample_size, len(chunks))
    rng = random.Random(seed)  # noqa: S311 - deterministic non-crypto sampling
    sampled_chunks = rng.sample(chunks, k=active_sample_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=60.0) as client:
        with output_path.open("w", encoding="utf-8") as handle:
            for index, chunk in enumerate(sampled_chunks, start=1):
                claims = _extract_claims_with_llm(
                    client=client,
                    api_key=api_key,
                    model=model,
                    chunk_text=chunk["chunk_text"],
                    max_claims_per_chunk=max_claims_per_chunk,
                )

                deduped_claims = _dedupe_claims(claims)
                record = {
                    "chunk_id": chunk["chunk_id"],
                    "source_id": chunk["source_id"],
                    "source_document_id": chunk["source_document_id"],
                    "document_checksum": chunk["document_checksum"],
                    "chunk_text": chunk["chunk_text"],
                    "gold_claims": [
                        {
                            "evidence_text": claim,
                            "normalized_claim_text": _normalize_text(claim),
                        }
                        for claim in deduped_claims
                    ],
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

                sys.stdout.write(
                    f"[{index}/{active_sample_size}] "
                    f"chunk_id={chunk['chunk_id']} gold_claims={len(deduped_claims)}\n",
                )

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)


def _load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Load chunk records from append-only JSONL chunk state."""
    if not chunks_path.exists():
        raise ValueError("chunks_path does not exist")

    chunks: list[dict[str, Any]] = []
    content = chunks_path.read_text(encoding="utf-8")
    for raw_line in content.splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        chunks.append(payload)
    return chunks


def _extract_claims_with_llm(
    *,
    client: httpx.Client,
    api_key: str,
    model: str,
    chunk_text: str,
    max_claims_per_chunk: int,
) -> list[str]:
    """Call an LLM to produce sentence-level claim candidates for one chunk."""
    prompt = (
        "Extract policy-relevant factual/normative claims from the text. "
        "Return only claims that are exact substrings from the input text. "
        "Return JSON object with key claims, where claims is an array of strings. "
        f"Limit to at most {max_claims_per_chunk} claims."
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise policy claim extraction assistant.",
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
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    claims = parsed.get("claims", [])
    if not isinstance(claims, list):
        return []
    return [str(item).strip() for item in claims if str(item).strip()]


def _normalize_text(value: str) -> str:
    """Normalize text for deterministic comparison and persistence."""
    return " ".join(value.split())


def _dedupe_claims(claims: list[str]) -> list[str]:
    """Dedupe claims by normalized text while preserving input order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        normalized = _normalize_text(claim)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(claim)
    return deduped


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for validation dataset generation script."""
    parser = argparse.ArgumentParser(
        prog="claims-eval-build-dataset",
        description="Build claim validation dataset using LLM labels.",
    )
    parser.add_argument(
        "--chunks-path",
        required=True,
        help="Path to chunk JSONL produced by analysis chunking.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to write labeled dataset JSONL.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Number of chunks to label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Chat completion model for claim labeling.",
    )
    parser.add_argument(
        "--max-claims-per-chunk",
        type=int,
        default=8,
        help="Maximum claims returned per chunk annotation call.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between LLM calls to reduce request pressure.",
    )
    return parser


def main() -> int:
    """Run dataset-generation script from CLI arguments."""
    parser = _build_parser()
    args = parser.parse_args()
    run(
        chunks_path=Path(args.chunks_path),
        output_path=Path(args.output_path),
        sample_size=args.sample_size,
        seed=args.seed,
        model=args.model,
        max_claims_per_chunk=args.max_claims_per_chunk,
        sleep_seconds=args.sleep_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
