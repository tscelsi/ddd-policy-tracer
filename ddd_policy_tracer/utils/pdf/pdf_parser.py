"""PDF parser using pymupdf4llm for layout parsing and custom logic to convert to markdown."""

import json
import logging
from typing import Any

import pymupdf4llm  # type: ignore
from pymupdf import pymupdf

from ddd_policy_tracer.utils.pdf.analyse_fonts import detect_paragraph_font_size

logger = logging.getLogger(__name__)

TOP_MARGIN = 50
BOTTOM_MARGIN = 50


class PdfParser:
    """Parser for PDFs, using pymupdf4llm for layout parsing and custom logic."""

    def __init__(
        self,
        min_font_size: int | float | None = None,
        top_margin: int = TOP_MARGIN,
        bottom_margin: int = BOTTOM_MARGIN,
    ) -> None:
        """Initialise the PDF parser with parameters for filtering text elements."""
        self.min_font_size = min_font_size
        self.top_margin = top_margin
        self.bottom_margin = bottom_margin

    def parse(self, pdf_bytes: bytes) -> str:
        """Parse the PDF bytes and return the extracted text as markdown."""
        doc = pymupdf.Document(stream=pdf_bytes)
        if self.min_font_size is None:
            p_font_size = detect_paragraph_font_size(doc)
            logger.debug(
                f"auto-detected p font size:{p_font_size}, set min_font_size:{p_font_size - 1}",
            )
            self.min_font_size = (
                p_font_size - 1
            )  # subtract 1 to avoid excluding paragraph text due to rounding
        # get all the goodness of pymupdf4llm's layout parsing while we have it available, then
        # convert to markdown with our custom parser for better fidelity to the original document
        # structure.
        json_str = pymupdf4llm.to_json(doc)  # type: ignore
        data = json.loads(json_str)
        md_text = self._parse_json(data)
        return md_text

    def _parse_json(self, data: dict[str, Any]) -> str:
        markdown_parts: list[str] = []

        for page in data["pages"]:
            page_height = page["height"]

            for box in page["boxes"]:
                # Skip non-text boxes
                if not box["textlines"]:
                    continue

                boxclass = box.get("boxclass", "")

                # Optional: skip headers/footers by box class
                if boxclass in {"caption", "page-footer", "footnote"}:
                    logger.debug(f"[skip] box with class {boxclass}")
                    continue

                box_lines: list[str] = []

                for line in box["textlines"]:
                    line_text: list[str] = []

                    bbox = line["bbox"]
                    y0 = bbox[1]
                    y1 = bbox[3]

                    # Skip top/bottom page regions
                    if y0 < self.top_margin:
                        logger.debug(f"[skip] line at y={y0} (above top margin {self.top_margin})")
                        continue

                    if y1 > page_height - self.bottom_margin:
                        logger.debug(
                            f"[skip] line at y={y1} (below bottom margin {page_height - self.bottom_margin})",
                        )
                        continue

                    for span in line["spans"]:
                        # filter small font sizes.
                        # keep them if in a list
                        if span["size"] < self.min_font_size and boxclass != "list-item":
                            logger.debug(f"[skip] {span['text']} (font size {span['size']})")
                            continue

                        line_text.append(span["text"])

                    combined = "".join(line_text).strip()

                    if combined:
                        box_lines.append(combined)

                if not box_lines:
                    continue

                # Simple markdown formatting based on box type
                if boxclass == "section-header":
                    markdown_parts.append("## " + " ".join(box_lines))
                else:
                    markdown_parts.append(" ".join(box_lines))

                markdown_parts.append("")

        return "\n".join(markdown_parts)
