"""Font analysis utilities for PDF documents."""

import sys
from collections import Counter
from typing import cast

import pymupdf


def detect_paragraph_font_size(doc: pymupdf.Document) -> float:
    """Detect the most common font size in the document as the paragraph font size."""
    c = Counter[float]()
    for page in doc:
        blocks = page.get_text("dict", flags=11)["blocks"]
        for b in blocks:
            for l in b["lines"]:
                for s in l["spans"]:
                    rounded_size = round(cast(float, s["size"]), 3)
                    c[rounded_size] += 1
    p_font_size = c.most_common(1)[0][0]
    # we subtract 1 from the most common font size so that we don't accidentally
    # exclude the paragraph size by rounding error. This shouldn't be a large enough
    # difference to include any footer text etc.
    return p_font_size


def flags_decomposer(flags):
    """Make font flags human readable."""
    l = []
    if flags & 2**0:
        l.append("superscript")
    if flags & 2**1:
        l.append("italic")
    if flags & 2**2:
        l.append("serifed")
    else:
        l.append("sans")
    if flags & 2**3:
        l.append("monospaced")
    else:
        l.append("proportional")
    if flags & 2**4:
        l.append("bold")
    return ", ".join(l)


def main() -> None:
    """Run a CLI tool to analyze fonts in a PDF document."""
    doc = pymupdf.open(sys.argv[1])
    page = doc[10]
    font_counter = Counter()
    size_counter = Counter()
    # read page text as a dictionary, suppressing extra spaces in CJK fonts
    for page in doc:
        blocks = page.get_text("dict", flags=11)["blocks"]
        for b in blocks:  # iterate through the text blocks
            for l in b["lines"]:  # iterate through the text lines
                for s in l["spans"]:  # iterate through the text spans
                    # print("")
                    s_font = s["font"]
                    s_flags = flags_decomposer(s["flags"])
                    s_size = s["size"]
                    s_color = s["color"]
                    font_counter[s_font] += 1
                    size_counter[s_size] += 1
                    # print(f"Text: '{s['text']}'")  # simple print of text
                    # print(f"Font: '{s_font}' ({s_flags}), size {s_size}, color #{s_color:06x}")

    print(font_counter)
    print(size_counter)


if __name__ == "__main__":
    main()
