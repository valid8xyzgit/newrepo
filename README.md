# RAG Pipeline Utilities

This repository packages a practical extraction pipeline for converting
layout-rich PDF documents (such as the 23 MB *Tax Summary 2025–26* flipbook)
into Markdown, CSV tables, and JSONL chunks that are ready for ingestion into a
retrieval augmented generation (RAG) stack built on top of Llama 3.3-70B.

## Features

- **Markdown extraction with heading preservation** using `pymupdf` +
  `markdownify`.
- **Table detection and export** to CSV and Markdown snippets with stable IDs
  (via native HTML parsing with a `pdfplumber` fallback).
- **Figure capture** with optional OCR (if `pytesseract` is installed).
- **Structure-aware chunking** that records heading paths, page spans, and adds
  `(p. …)` anchors for provenance.
- **JSONL serialisation** compatible with vector stores and retrievers.
- **CLI utility** for ad-hoc conversions of uploaded PDFs.

## Installation

The pipeline depends on a handful of Python packages:

```bash
pip install -r requirements.txt
```

If you prefer not to use `requirements.txt`, install the dependencies manually:

```bash
pip install pymupdf markdownify beautifulsoup4 pandas pdfplumber tabulate
```

Optional:

- `pytesseract` + the native Tesseract binary (for OCR of flowchart images).

## Quickstart: Run the Pipeline on a PDF

1. (Optional) Create helper directories that keep uploads and outputs tidy:

   ```bash
   mkdir -p uploads outputs
   ```

2. Copy or upload a PDF into `uploads/` (for example `uploads/tax-summary.pdf`).

3. Execute the CLI. The command below writes artefacts into
   `outputs/tax-summary/` and tags the chunks with `doc_id="tax-summary-2025"`:

   ```bash
   python -m rag_pipeline uploads/tax-summary.pdf \
       --output-dir outputs/tax-summary \
       --doc-id tax-summary-2025
   ```

4. Inspect the generated files:

   - `outputs/tax-summary/document.md` – Markdown with headings, bold text,
     table/figure placeholders, and page anchors.
   - `outputs/tax-summary/tables/*.csv` – machine-readable tables (if any were
     present).
   - `outputs/tax-summary/figures/*` – extracted images (OCR text is stored in
     the JSONL chunk metadata when available).
   - `outputs/tax-summary/chunks.jsonl` – chunk objects ready to embed and
     index.

To process a different document, simply swap out the input/output paths above.

## Run the Hosted Web Service

If you prefer to serve the pipeline behind a web UI or deploy it to a hosting
platform (e.g. Hugging Face Spaces, Fly.io, Render, or an internal VM), start
the FastAPI app defined in `app.py`:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Once running you can:

- Visit `http://localhost:8000/` for a simple upload form that returns a ZIP
  bundle of Markdown, tables, figures, and JSONL chunks.
- Send an HTTP `POST` request to `/process` with a multipart body containing
  `file` (the PDF) and `doc_id` (string). The response is a downloadable ZIP
  archive suitable for automated test harnesses or integrations.
- Poll `/health` for readiness checks when deploying to container platforms.

## Programmatic Usage

```python
from rag_pipeline import process_pdf_for_rag

result = process_pdf_for_rag(
    "uploads/tax-summary.pdf",
    "outputs/tax-summary",
    doc_id="tax-summary-2025",
)

print(result.chunks_path)
```

## Running the Automated Test

The repository ships with a synthetic PDF excerpt in `sample_docs/` and a
pytest suite that exercises the end-to-end pipeline:

```bash
pytest
```

The test confirms that Markdown, table CSVs, and JSONL chunks are produced and
contain the expected metadata.

### Using Your Own PDF in the Test Suite

You can reuse the integration test to validate another document without
touching the code. Point the environment variable `RAG_PIPELINE_TEST_PDF` at the
file you want to process and run `pytest`:

```bash
RAG_PIPELINE_TEST_PDF=uploads/your-document.pdf pytest -k test_process_pdf_creates_expected_artifacts
```

The test will emit all artefacts into a temporary directory (displayed in the
pytest output) so you can inspect the Markdown, tables, and JSONL chunks created
from your own upload.
