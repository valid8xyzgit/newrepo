from pathlib import Path
import json
import os
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_pipeline import process_pdf_for_rag


@pytest.fixture(scope="module")
def pdf_under_test() -> Path:
    """Return the PDF to exercise in the integration test.

    Users can point the test at an arbitrary document by exporting the
    ``RAG_PIPELINE_TEST_PDF`` environment variable before invoking ``pytest``.
    When the variable is unset we fall back to the bundled synthetic sample
    excerpt to keep CI stable.
    """

    custom_pdf = os.getenv("RAG_PIPELINE_TEST_PDF")
    if custom_pdf:
        pdf_path = Path(custom_pdf).expanduser()
        if not pdf_path.exists():
            pytest.skip(f"custom PDF not found: {pdf_path}")
        return pdf_path

    pdf_path = Path("sample_docs/sample_tax_excerpt.pdf")
    if not pdf_path.exists():
        pytest.skip("sample PDF not generated")
    return pdf_path


@pytest.fixture(scope="module")
def is_custom_pdf() -> bool:
    return bool(os.getenv("RAG_PIPELINE_TEST_PDF"))


def test_process_pdf_creates_expected_artifacts(
    tmp_path: Path, pdf_under_test: Path, is_custom_pdf: bool
) -> None:
    doc_id = pdf_under_test.stem.replace(" ", "-")
    result = process_pdf_for_rag(pdf_under_test, tmp_path, doc_id=doc_id)

    assert result.markdown_path.exists()
    assert result.chunks_path.exists()

    with result.markdown_path.open("r", encoding="utf-8") as fh:
        markdown = fh.read()
    if not is_custom_pdf:
        assert "Small Business CGT Concessions" in markdown
        assert "[Table P0001-T01]" in markdown

    with result.chunks_path.open("r", encoding="utf-8") as fh:
        chunk_lines = [json.loads(line) for line in fh if line.strip()]

    assert any(chunk["type"] == "text" for chunk in chunk_lines)
    if not is_custom_pdf:
        assert any(chunk["type"] == "table" for chunk in chunk_lines)

    # ensure table CSV exists
    if result.tables:
        for table in result.tables:
            csv_path = tmp_path / Path(table.csv_relpath)
            assert csv_path.exists()

    # ensure chunk titles include heading context
    text_chunks = [chunk for chunk in chunk_lines if chunk["type"] == "text"]
    assert text_chunks
    assert all("(p" in chunk["text"] for chunk in text_chunks)
