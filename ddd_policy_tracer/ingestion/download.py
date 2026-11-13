from pathlib import Path

from ddd_policy_tracer.ingestion.adapters import DiscoveredDocument


async def download_source_documents(
    source_id: str,
    dir: Path,
    discovered_documents: list[DiscoveredDocument],
):
    save_dir = Path(f"{dir}/{source_id}")
