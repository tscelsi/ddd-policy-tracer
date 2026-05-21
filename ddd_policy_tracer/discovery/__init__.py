"""Public package exports for document acquisition workflows."""

from .cli import run_cli
from .domain import SourceDocument
from .service_layer import (
    AcquisitionReport,
    get_source_document_versions,
    ingest_source_documents,
)

__all__ = [
    "AcquisitionReport",
    "SourceDocument",
    "get_source_document_versions",
    "ingest_source_documents",
    "run_cli",
]
