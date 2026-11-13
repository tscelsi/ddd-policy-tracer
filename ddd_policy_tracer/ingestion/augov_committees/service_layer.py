"""Service layer for the Augov Committees discovery process."""

import logging
from pathlib import Path

from ddd_policy_tracer.ingestion.adapters import (
    DiskArtifactStore,
    FilesystemSourceDocumentRepository,
)
from ddd_policy_tracer.ingestion.augov_committees.get_documents import (
    get_submission_documents,
)

logging.basicConfig(level=logging.INFO)


def save_submission_documents_for_inquiry() -> None:
    """Fetches submission documents and stores them."""
    artifact_store = DiskArtifactStore(artifact_dir=Path("data/committee_artifacts"))
    doc_repo = FilesystemSourceDocumentRepository(state_path="data/committee_documents.jsonl")
    docs = get_submission_documents(
        submissions_url="https://www.aph.gov.au/Parliamentary_Business/Committees/Senate/Environment_and_Communications/CRA48P/Submissions",
        artifact_store=artifact_store,
    )
    for doc in docs:
        doc_repo.add_version(doc)


if __name__ == "__main__":
    save_submission_documents_for_inquiry()
