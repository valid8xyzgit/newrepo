"""Utilities for converting layout-rich PDFs into RAG-friendly artifacts.

This module provides two main entry points:

* :func:`build_llm_ready_pipeline_description` – returns the Markdown-formatted
  workflow overview originally requested in the chat transcript.
* :func:`process_pdf_for_rag` – executes a concrete extraction pipeline that
  converts a PDF into Markdown, table CSVs, optional figure OCR text, and a
  JSONL file containing ready-to-index chunks with metadata tailored for
  Llama 3.3-70B retrieval augmented generation workloads.

The implementation favours readability and composability so teams can adapt the
steps to their infrastructure (e.g., swapping out the extractor or embedding
model).  Each helper is designed to be testable in isolation and the
``process_pdf_for_rag`` function exposes a succinct return object that callers
can inspect programmatically or use to trigger downstream tasks such as
embedding generation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import fitz  # type: ignore[import-not-found]
import pandas as pd
from bs4 import BeautifulSoup
from markdownify import markdownify as to_markdown
import pdfplumber

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TableExtraction:
    """Metadata describing a table extracted from the PDF."""

    table_id: str
    page: int
    markdown: str
    csv_relpath: str
    heading_path: List[str] = field(default_factory=list)


@dataclass(slots=True)
class FigureExtraction:
    """Metadata describing a figure/flowchart and optional OCR text."""

    figure_id: str
    page: int
    image_relpath: str
    ocr_text: str = ""
    heading_path: List[str] = field(default_factory=list)


@dataclass(slots=True)
class Chunk:
    """Structured representation of a chunk ready for indexing."""

    doc_id: str
    chunk_id: str
    type: str
    heading_path: List[str]
    page_start: int
    page_end: int
    title: str
    text: str
    table_id: Optional[str] = None
    figure_id: Optional[str] = None
    source_relpath: Optional[str] = None

    def asdict(self) -> dict:
        """Return the JSON-serialisable representation of the chunk."""

        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "type": self.type,
            "heading_path": self.heading_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "title": self.title,
            "table_id": self.table_id,
            "figure_id": self.figure_id,
            "source_relpath": self.source_relpath,
            "text": self.text,
        }


@dataclass(slots=True)
class PipelineResult:
    """Container returned by :func:`process_pdf_for_rag`."""

    markdown_path: Path
    chunks_path: Path
    tables: List[TableExtraction]
    figures: List[FigureExtraction]
    chunks: List[Chunk]


# ---------------------------------------------------------------------------
# Human-readable pipeline description
# ---------------------------------------------------------------------------


def build_llm_ready_pipeline_description() -> str:
    """Return the step-by-step pipeline description for preparing the PDF."""

    return (
        "1. **Extract the content while preserving structure**\n"
        "   - Use a layout-aware parser (e.g., docling, unstructured, pdfplumber)\n"
        "     to emit Markdown that keeps headings, lists, and **bold** text.\n"
        "   - Export tables separately as CSV/Markdown using tools like camelot or\n"
        "     tabula; maintain table identifiers for cross-referencing.\n"
        "   - Capture flowchart logic: scrape vector text when possible, or OCR\n"
        "     embedded images and convert them to textual adjacency lists and\n"
        "     optional Mermaid diagrams.\n\n"
        "2. **Normalize and enrich the extracted data**\n"
        "   - Retain Markdown emphasis and append page anchors (e.g., ``(p. 125)``)\n"
        "     to paragraphs for provenance tracking.\n"
        "   - Maintain hierarchical heading paths, table/figure identifiers, and\n"
        "     other metadata that will accompany each downstream chunk.\n\n"
        "3. **Apply structure-aware chunking**\n"
        "   - Narrative sections: create 250–400 token chunks with 40–60 token\n"
        "     overlaps, aligned with paragraph and heading boundaries.\n"
        "   - Tables: keep both full-table chunks (caption + sample rows) and\n"
        "     logical row slices, always duplicating header rows.\n"
        "   - Flowcharts: produce plain-language summaries plus machine-readable\n"
        "     edge lists or Mermaid blocks for downstream retrieval.\n\n"
        "4. **Serialize chunks into JSONL with rich metadata**\n"
        "   - Example schema fields: ``doc_id``, ``chunk_id``, ``type``\n"
        "     (text/table/figure), ``heading_path``, ``page_start``/``page_end``,\n"
        "     ``title``, ``table_id``/``figure_id`` (when applicable),\n"
        "     ``source_relpath``, and ``text``.\n\n"
        "5. **Embed using a dedicated encoder and index the data**\n"
        "   - Generate embeddings with a strong open-source model (e.g.,\n"
        "     BGE-M3, E5-Mistral, NV-Embed v2) and load them into a dense vector\n"
        "     store.\n"
        "   - Build a complementary BM25/keyword index for exact matches on\n"
        "     section numbers, acronyms, and table headers.\n\n"
        "6. **Tune retrieval for Llama 3.3-70B answer generation**\n"
        "   - Optionally rewrite queries to expand acronyms and surface implicit\n"
        "     constraints before retrieval.\n"
        "   - Retrieve a broad candidate set (e.g., top 12–20 dense + 20 BM25),\n"
        "     rerank to the best 5–8, and instruct the model to prioritise tables\n"
        "     for numeric questions and flowcharts for process queries.\n\n"
        "7. **Evaluate the pipeline and add guardrails**\n"
        "   - Generate a synthetic evaluation set (30–50 queries) to measure hit\n"
        "     rate, answer faithfulness, and citation accuracy.\n"
        "   - Implement an \"unanswerable\" fallback when relevant context is not\n"
        "     retrieved with sufficient confidence.\n\n"
        "8. **Adopt a practical, automation-friendly toolchain**\n"
        "   - Automate PDF → Markdown/CSV conversion, OCR, chunking, and JSONL\n"
        "     export with Python tooling (docling/unstructured + pdfplumber,\n"
        "     camelot/tabula, Tesseract/PaddleOCR, etc.).\n"
        "   - Embed and index with your preferred vector database and keyword\n"
        "     engine, then orchestrate retrieval + Llama 3.3-70B generation.\n\n"
        "9. **Special handling for the Tax Summary 2025–26 PDF**\n"
        "   - Preserve bold emphasis as a signal of key definitions and consider\n"
        "     tagging chunks containing significant bold text.\n"
        "   - Provide both overview and row-level chunks for large tables, and\n"
        "     accompany flowchart text with adjacency lists to improve recall."
    )


# ---------------------------------------------------------------------------
# PDF → Markdown/Tables/Figures extraction helpers
# ---------------------------------------------------------------------------


def _estimate_token_count(text: str) -> int:
    """Rudimentary word-based token estimator suitable for chunk sizing."""

    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, math.ceil(len(cleaned.split()) * 0.9))


@dataclass(slots=True)
class _Paragraph:
    text: str
    heading_path: List[str]
    page_start: int
    page_end: int
    token_count: int


def _page_mark(page: int) -> str:
    return f"<!-- page:{page} -->"


def _extract_tables(
    soup: BeautifulSoup,
    page_number: int,
    tables_dir: Path,
    plumber_page: Optional["pdfplumber.page.Page"] = None,
) -> List[TableExtraction]:
    """Extract tables from the page soup, returning metadata objects."""

    tables: List[TableExtraction] = []
    html_tables = list(soup.find_all("table"))
    for index, table in enumerate(html_tables, start=1):
        table_id = f"P{page_number:04d}-T{index:02d}"
        dataframes = pd.read_html(str(table))
        if not dataframes:
            table.decompose()
            continue
        df = dataframes[0]
        csv_relpath = Path("tables") / f"{table_id}.csv"
        csv_path = tables_dir / csv_relpath.name
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        table_markdown = df.to_markdown(index=False)

        placeholder = soup.new_tag("p")
        placeholder.string = f"[Table {table_id}](tables/{csv_relpath.name})"
        table.replace_with(placeholder)

        tables.append(
            TableExtraction(
                table_id=table_id,
                page=page_number,
                markdown=table_markdown,
                csv_relpath=str(csv_relpath),
            )
        )
    if plumber_page is not None and not tables:
        tables.extend(
            _extract_tables_from_pdfplumber(
                soup=soup,
                plumber_page=plumber_page,
                page_number=page_number,
                tables_dir=tables_dir,
                start_index=len(html_tables) + 1,
            )
        )
    return tables


def _extract_tables_from_pdfplumber(
    *,
    soup: BeautifulSoup,
    plumber_page: "pdfplumber.page.Page",
    page_number: int,
    tables_dir: Path,
    start_index: int,
) -> List[TableExtraction]:
    """Fallback table extraction using pdfplumber when HTML lacks <table> nodes."""

    tables: List[TableExtraction] = []
    raw_tables = plumber_page.extract_tables() or []
    for offset, raw_table in enumerate(raw_tables, start=start_index):
        if not raw_table:
            continue
        header = raw_table[0]
        body = raw_table[1:] if len(raw_table) > 1 else []
        if any(cell is not None for cell in header):
            df = pd.DataFrame(body, columns=header)
        else:
            df = pd.DataFrame(raw_table)
        df = df.fillna("")

        table_id = f"P{page_number:04d}-T{offset:02d}"
        csv_relpath = Path("tables") / f"{table_id}.csv"
        csv_path = tables_dir / csv_relpath.name
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        table_markdown = df.to_markdown(index=False)

        placeholder = soup.new_tag("p")
        placeholder.string = f"[Table {table_id}](tables/{csv_relpath.name})"
        parent = soup.body or soup
        parent.append(placeholder)

        tables.append(
            TableExtraction(
                table_id=table_id,
                page=page_number,
                markdown=table_markdown,
                csv_relpath=str(csv_relpath),
            )
        )
    return tables


def _extract_images(
    doc: fitz.Document,
    page: fitz.Page,
    soup: BeautifulSoup,
    page_number: int,
    figures_dir: Path,
) -> List[FigureExtraction]:
    """Save images from the page and annotate the soup with placeholders."""

    figures: List[FigureExtraction] = []
    image_tags = list(soup.find_all("img"))
    tag_iter = iter(image_tags)

    for index, image in enumerate(page.get_images(full=True), start=1):
        xref = image[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image.get("image")
        if not image_bytes:
            continue
        extension = base_image.get("ext", "png")
        figure_id = f"P{page_number:04d}-F{index:02d}"
        image_relpath = Path("figures") / f"{figure_id}.{extension}"
        image_path = figures_dir / image_relpath.name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)

        placeholder = None
        try:
            placeholder = next(tag_iter)
        except StopIteration:
            placeholder = soup.new_tag("p")
            soup.body.append(placeholder)
        placeholder.name = "p"
        placeholder.string = f"[Figure {figure_id}](figures/{image_relpath.name})"

        ocr_text = ""
        try:
            from PIL import Image  # type: ignore[import-not-found]
            import pytesseract  # type: ignore[import-not-found]

            with Image.open(image_path) as img:  # type: ignore[attr-defined]
                ocr_text = pytesseract.image_to_string(img).strip()
        except Exception:
            ocr_text = ""

        figures.append(
            FigureExtraction(
                figure_id=figure_id,
                page=page_number,
                image_relpath=str(image_relpath),
                ocr_text=ocr_text,
            )
        )

    # Remove any leftover <img> tags to avoid raw data URIs in Markdown
    for remaining in tag_iter:
        remaining.decompose()

    return figures


def _extract_pdf_to_markdown(
    pdf_path: Path,
    output_dir: Path,
) -> tuple[str, List[TableExtraction], List[FigureExtraction]]:
    """Convert ``pdf_path`` into Markdown and capture table/figure metadata."""

    doc = fitz.open(pdf_path)
    plumber_doc = pdfplumber.open(pdf_path)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_tables: List[TableExtraction] = []
    all_figures: List[FigureExtraction] = []
    markdown_pages: List[str] = []

    for page_index, page in enumerate(doc, start=1):
        html = page.get_text("html")
        soup = BeautifulSoup(html, "html.parser")
        plumber_page = plumber_doc.pages[page_index - 1]
        page_tables = _extract_tables(soup, page_index, tables_dir, plumber_page=plumber_page)
        all_tables.extend(page_tables)

        page_figures = _extract_images(doc, page, soup, page_index, figures_dir)
        all_figures.extend(page_figures)

        markdown = to_markdown(str(soup), heading_style="ATX")
        markdown_pages.append(f"{_page_mark(page_index)}\n\n{markdown.strip()}".strip())

    plumber_doc.close()
    doc.close()

    return "\n\n".join(markdown_pages), all_tables, all_figures


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------


def _iter_paragraphs(markdown_text: str) -> Iterator[_Paragraph]:
    """Yield paragraphs with heading context and page spans."""

    heading_stack: List[str] = []
    current_lines: List[str] = []
    current_pages: List[int] = []
    page_number = 0

    def flush_current() -> Iterator[_Paragraph]:
        nonlocal current_lines, current_pages
        if not current_lines:
            return iter(())
        text = "\n".join(current_lines).strip()
        if not text:
            current_lines = []
            current_pages = []
            return iter(())
        token_count = _estimate_token_count(text)
        page_start = min(current_pages) if current_pages else page_number
        page_end = max(current_pages) if current_pages else page_number
        paragraph = _Paragraph(
            text=text,
            heading_path=list(heading_stack),
            page_start=page_start,
            page_end=page_end,
            token_count=token_count,
        )
        current_lines = []
        current_pages = []
        return iter((paragraph,))

    lines = markdown_text.splitlines()
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line:
            yield from flush_current()
            continue
        if line.startswith("<!-- page:") and line.endswith("-->"):
            yield from flush_current()
            try:
                page_number = int(line.split(":", 1)[1].strip(" -<>"))
            except ValueError:
                page_number += 1
            continue
        if line.startswith("#"):
            yield from flush_current()
            level = len(line) - len(line.lstrip("#"))
            heading_text = line[level:].strip()
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(heading_text)
            yield _Paragraph(
                text=line.strip(),
                heading_path=list(heading_stack),
                page_start=page_number,
                page_end=page_number,
                token_count=_estimate_token_count(heading_text),
            )
            continue

        current_lines.append(line.strip())
        current_pages.append(page_number)

    yield from flush_current()


def _derive_title(heading_path: Sequence[str], paragraph_text: str) -> str:
    """Generate a concise title for a chunk."""

    if heading_path:
        return heading_path[-1]
    words = paragraph_text.split()
    return " ".join(words[: min(len(words), 12)])


def _chunk_text_paragraphs(
    doc_id: str,
    paragraphs: Iterable[_Paragraph],
    chunk_token_target: int,
    chunk_overlap_tokens: int,
    source_relpath: str,
) -> List[Chunk]:
    """Convert an iterable of paragraphs into chunk metadata objects."""

    chunks: List[Chunk] = []
    buffer: List[_Paragraph] = []
    buffer_tokens = 0
    chunk_index = 1

    def flush() -> None:
        nonlocal buffer, buffer_tokens, chunk_index
        if not buffer:
            return
        text = "\n\n".join(p.text for p in buffer).strip()
        if not text:
            buffer = []
            buffer_tokens = 0
            return
        heading_path = buffer[-1].heading_path
        page_numbers = [
            value
            for paragraph in buffer
            for value in (paragraph.page_start, paragraph.page_end)
            if value
        ]
        page_start = min(page_numbers) if page_numbers else 0
        page_end = max(page_numbers) if page_numbers else 0
        title = _derive_title(heading_path, buffer[-1].text)
        anchor = ""
        if page_start and page_end:
            if page_start == page_end:
                anchor = f"(p. {page_start})"
            else:
                anchor = f"(pp. {page_start}–{page_end})"
        text_with_anchor = f"{text}\n\n{anchor}".strip() if anchor else text
        chunk = Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}-text-{chunk_index:05d}",
            type="text",
            heading_path=heading_path,
            page_start=page_start,
            page_end=page_end,
            title=title,
            text=text_with_anchor,
            source_relpath=source_relpath,
        )
        chunks.append(chunk)
        chunk_index += 1

        overlap: List[_Paragraph] = []
        overlap_tokens = 0
        for paragraph in reversed(buffer):
            overlap_tokens += paragraph.token_count
            overlap.append(paragraph)
            if overlap_tokens >= chunk_overlap_tokens:
                break
        buffer = list(reversed(overlap))
        buffer_tokens = sum(p.token_count for p in buffer)

    for paragraph in paragraphs:
        # Skip placeholders for tables/figures – handled separately.
        if paragraph.text.startswith("[Table ") or paragraph.text.startswith("[Figure "):
            flush()
            buffer = []
            buffer_tokens = 0
            continue

        tentative = buffer_tokens + paragraph.token_count
        if buffer and tentative > chunk_token_target:
            flush()
        buffer.append(paragraph)
        buffer_tokens += paragraph.token_count

    flush()
    return chunks


def _attach_headings_to_media(
    media: List[TableExtraction] | List[FigureExtraction],
    paragraphs: Iterable[_Paragraph],
) -> None:
    """Mutate ``media`` items so they inherit the nearest heading context."""

    media_lookup = {item.table_id if isinstance(item, TableExtraction) else item.figure_id: item for item in media}
    for paragraph in paragraphs:
        text = paragraph.text
        if text.startswith("[Table "):
            identifier = text.split("[Table ", 1)[1].split("]", 1)[0]
            item = media_lookup.get(identifier)
            if isinstance(item, TableExtraction):
                item.heading_path = paragraph.heading_path
        elif text.startswith("[Figure "):
            identifier = text.split("[Figure ", 1)[1].split("]", 1)[0]
            item = media_lookup.get(identifier)
            if isinstance(item, FigureExtraction):
                item.heading_path = paragraph.heading_path


# ---------------------------------------------------------------------------
# Public orchestration entry point
# ---------------------------------------------------------------------------


def process_pdf_for_rag(
    pdf_path: Path | str,
    output_dir: Path | str,
    *,
    doc_id: str,
    chunk_token_target: int = 350,
    chunk_overlap_tokens: int = 60,
) -> PipelineResult:
    """Extract Markdown, tables, figures, and JSONL chunks from ``pdf_path``.

    Parameters
    ----------
    pdf_path:
        Location of the PDF to process.
    output_dir:
        Directory that will contain the Markdown, table CSVs, figures, and the
        resulting ``chunks.jsonl`` file.
    doc_id:
        Stable identifier used in chunk metadata (e.g., ``"tax-summary-2025"``).
    chunk_token_target / chunk_overlap_tokens:
        Chunk sizing heuristics, expressed in pseudo-token counts (word-based
        approximation).  The defaults mirror the recommendations from the
        original requirement.

    Returns
    -------
    :class:`PipelineResult`
        Includes paths to the produced artifacts along with in-memory metadata
        for further automation or testing.
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_text, tables, figures = _extract_pdf_to_markdown(pdf_path, output_dir)
    markdown_path = output_dir / "document.md"
    markdown_path.write_text(markdown_text, encoding="utf-8")

    paragraphs = list(_iter_paragraphs(markdown_text))
    _attach_headings_to_media(tables, paragraphs)
    _attach_headings_to_media(figures, paragraphs)

    text_chunks = _chunk_text_paragraphs(
        doc_id=doc_id,
        paragraphs=(p for p in paragraphs if not (p.text.startswith("[Table ") or p.text.startswith("[Figure "))),
        chunk_token_target=chunk_token_target,
        chunk_overlap_tokens=chunk_overlap_tokens,
        source_relpath=markdown_path.name,
    )

    table_chunks = [
        Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}-table-{index:05d}",
            type="table",
            heading_path=table.heading_path,
            page_start=table.page,
            page_end=table.page,
            title=f"Table {table.table_id}",
            text=table.markdown,
            table_id=table.table_id,
            source_relpath=table.csv_relpath,
        )
        for index, table in enumerate(tables, start=1)
    ]

    figure_chunks = [
        Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}-figure-{index:05d}",
            type="figure",
            heading_path=figure.heading_path,
            page_start=figure.page,
            page_end=figure.page,
            title=f"Figure {figure.figure_id}",
            text=figure.ocr_text or f"Figure available at {figure.image_relpath}",
            figure_id=figure.figure_id,
            source_relpath=figure.image_relpath,
        )
        for index, figure in enumerate(figures, start=1)
    ]

    chunks = [*text_chunks, *table_chunks, *figure_chunks]
    chunks_path = output_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.asdict(), ensure_ascii=False) + "\n")

    return PipelineResult(
        markdown_path=markdown_path,
        chunks_path=chunks_path,
        tables=tables,
        figures=figures,
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# CLI convenience for manual testing with uploaded PDFs
# ---------------------------------------------------------------------------


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a PDF into Markdown, tables, and JSONL chunks for RAG.",
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF to process")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rag_output"),
        help="Directory where artifacts will be written",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default="uploaded-document",
        help="Identifier injected into chunk metadata",
    )
    parser.add_argument(
        "--chunk-target",
        type=int,
        default=350,
        help="Approximate token target for each chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=60,
        help="Approximate overlap tokens between adjacent chunks",
    )

    args = parser.parse_args()

    result = process_pdf_for_rag(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        doc_id=args.doc_id,
        chunk_token_target=args.chunk_target,
        chunk_overlap_tokens=args.chunk_overlap,
    )

    print(f"Markdown written to: {result.markdown_path}")
    print(f"Chunks JSONL written to: {result.chunks_path}")
    if result.tables:
        print("Extracted tables:")
        for table in result.tables:
            print(f"  - {table.table_id} (page {table.page}) -> {table.csv_relpath}")
    else:
        print("No tables detected in the document.")
    if result.figures:
        print("Extracted figures:")
        for figure in result.figures:
            print(
                f"  - {figure.figure_id} (page {figure.page}) -> {figure.image_relpath}"
                + (" [OCR]" if figure.ocr_text else "")
            )
    else:
        print("No figures detected in the document.")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _cli()


__all__ = [
    "Chunk",
    "FigureExtraction",
    "PipelineResult",
    "TableExtraction",
    "build_llm_ready_pipeline_description",
    "process_pdf_for_rag",
]
