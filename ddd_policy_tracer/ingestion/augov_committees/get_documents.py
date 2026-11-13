"""Module for fetching documents from the Augov Senate Committees website."""

import logging
from typing import cast

import httpx
from bs4 import BeautifulSoup

from ddd_policy_tracer.ingestion.adapters import DiskArtifactStore
from ddd_policy_tracer.ingestion.domain import (
    SourceDocument,
    compute_checksum,
    normalize_source_document_id,
)
from ddd_policy_tracer.utils.pdf.pdf_parser import PdfParser
from ddd_policy_tracer.utils.time_helpers import utc_now_isoformat

logger = logging.getLogger(__name__)
BASE_URL = "https://www.aph.gov.au"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:143.0) Gecko/20100101 Firefox/143.0",
}


def _get_submission_links(submissions_url: str) -> dict[str, str]:
    """Returns a dictionary mapping submitter names to their submission document links."""
    res = httpx.get(submissions_url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find("table", {"class": "rgMasterTable"})
    if not table:
        raise ValueError("Could not find table with class 'rgMasterTable'")
    submission_links: dict[str, str] = {}
    for row in table.find_all("tr", class_=["rgRow", "rgAltRow"]):
        tag_with_name = row.find("strong")
        submitted_by = tag_with_name.text.strip() if tag_with_name else "Unknown"
        first_anchor = row.find("a", {"class": "fileLink", "href": True})
        if first_anchor:
            submission_links[submitted_by] = BASE_URL + cast(str, first_anchor["href"])
    return submission_links


def get_submission_documents(
    submissions_url: str,
    artifact_store: DiskArtifactStore,
) -> list[SourceDocument]:
    """Fetches submissions for a committee inquiry.

    URL should look something like: https://..../Committees/Senate/Environment_and_Communications/CRA48P/Submissions
    """
    pdf_links = _get_submission_links(submissions_url)
    logger.info(f"Found {len(pdf_links)} submission documents to fetch.")
    documents: list[SourceDocument] = []
    for submitted_by, link in pdf_links.items():
        res = httpx.get(link, headers=headers)
        if res.status_code != 200:
            logger.warning(
                f"Failed to fetch document for {submitted_by} at {link}: {res.status_code}",
            )
            continue
        parser = PdfParser()  # new parser each iteration to detect font sizes properly
        pdf_bytes = res.content
        markdown = parser.parse(pdf_bytes)
        now = utc_now_isoformat()
        source_document_id = normalize_source_document_id(link)
        content_type = "application/pdf"
        raw_content_ref = artifact_store.store(
            source_document_id=source_document_id,
            content=pdf_bytes,
            filetype=content_type,
        )
        logger.info(f"Fetched and parsed document for {submitted_by} at {link}")
        documents.append(
            SourceDocument(
                source_id="committees",
                source_document_id=normalize_source_document_id(link),
                source_url=link,
                published_at=None,
                retrieved_at=now,
                checksum=compute_checksum(pdf_bytes),
                normalized_text=markdown,
                raw_content_ref=raw_content_ref,
                content_type=content_type,
                created_at=now,
                updated_at=now,
                metadata={
                    "submitted_by": submitted_by,
                },
            ),
        )
    return documents
