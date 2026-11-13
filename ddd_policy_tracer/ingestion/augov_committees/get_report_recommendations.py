import httpx
from bs4 import BeautifulSoup

from ddd_policy_tracer.__DEPRECATED_analysis.enrichment.chunks.chunking import _build_chunk_id
from ddd_policy_tracer.__DEPRECATED_analysis.enrichment.chunks.models import DocumentChunk
from ddd_policy_tracer.ingestion.domain import compute_checksum

url = "https://www.aph.gov.au/Parliamentary_Business/Committees/Senate/Environment_and_Communications/CRA48P/Report/List_of_recommendations"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:143.0) Gecko/20100101 Firefox/143.0",
}


def get_report_recommendations(url: str) -> list[DocumentChunk]:
    res = httpx.get(url, headers=headers)
    if res.status_code != 200:
        raise ValueError(f"Failed to fetch report recommendations page: {res.status_code}")
    soup = BeautifulSoup(res.text, "html.parser")
    recommendations: list[str] = []
    divs = soup.find_all("p", {"class": "RecommendationListText"})
    if not divs:
        raise ValueError("Could not find any divs with class 'RecommendationListText'")
    for div in divs:
        spans = div.find_all("span")
        recommendations.append(spans[-1].get_text())
    chunks: list[DocumentChunk] = []
    source_id = "committees"
    source_document_id = "report_recommendations"
    checksum = compute_checksum("".join(recommendations).encode("utf-8"))
    for i, rec in enumerate(recommendations):
        chunk = DocumentChunk(
            chunk_id=_build_chunk_id(
                source_id=source_id,
                source_document_id=source_document_id,
                checksum=checksum,
                chunk_index=i,
                start_char=0,
                end_char=0,
            ),
            source_id=source_id,
            source_document_id=source_document_id,
            document_checksum=checksum,
            chunk_index=i,
            start_char=0,
            end_char=0,
            chunk_text=rec,
        )
        chunks.append(chunk)
    return chunks
