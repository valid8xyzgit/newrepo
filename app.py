"""FastAPI service for running the RAG PDF processing pipeline."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from rag_pipeline import process_pdf_for_rag


app = FastAPI(
    title="PDF → RAG Pipeline",
    description=(
        "Upload a layout-rich PDF and receive Markdown, table CSVs, and JSONL "
        "chunks tailored for Llama 3.3-70B retrieval augmented generation workflows."
    ),
)


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>PDF → RAG Pipeline</title>
    <style>
      :root {
        color-scheme: light dark;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      body {
        margin: 2rem auto;
        max-width: 700px;
        line-height: 1.6;
        padding: 0 1.5rem;
      }
      header {
        margin-bottom: 1.5rem;
      }
      form {
        border: 1px solid #8884;
        border-radius: 0.75rem;
        padding: 1.5rem;
        display: grid;
        gap: 1rem;
      }
      label {
        display: block;
        font-weight: 600;
      }
      input[type="text"],
      input[type="file"] {
        width: 100%;
        padding: 0.5rem 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #8884;
      }
      button {
        justify-self: start;
        padding: 0.6rem 1.2rem;
        border-radius: 999px;
        border: none;
        background: #2563eb;
        color: white;
        font-weight: 600;
        cursor: pointer;
      }
      button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
      }
      footer {
        margin-top: 2rem;
        font-size: 0.9rem;
        color: #666;
      }
    </style>
  </head>
  <body>
    <header>
      <h1>PDF → RAG Pipeline</h1>
      <p>
        Upload a PDF and receive a ZIP bundle containing Markdown, table CSVs, figure metadata,
        and JSONL chunks that you can ingest into a RAG stack tuned for Llama 3.3-70B.
      </p>
    </header>
    <form id="upload-form" method="post" action="/process" enctype="multipart/form-data">
      <div>
        <label for="file">Select PDF</label>
        <input id="file" name="file" type="file" accept="application/pdf" required />
      </div>
      <div>
        <label for="doc_id">Document ID (used in chunk metadata)</label>
        <input id="doc_id" name="doc_id" type="text" value="uploaded-pdf" required />
      </div>
      <button type="submit">Process PDF</button>
    </form>
    <footer>
      <p>
        Prefer an automated workflow? Send a <code>POST</code> request to <code>/process</code> with a multipart
        body containing <code>file</code> (the PDF) and <code>doc_id</code> (string). The response is a ZIP archive
        with the full set of artefacts.
      </p>
    </footer>
    <script>
      const form = document.getElementById("upload-form");
      form.addEventListener("submit", () => {
        const button = form.querySelector("button");
        button.disabled = true;
        button.textContent = "Processing…";
      });
    </script>
  </body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve a minimal HTML interface for uploading PDFs."""

    return INDEX_HTML


@app.post("/process")
async def process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_id: str = Form(...),
):
    """Process an uploaded PDF and return a ZIP of generated artefacts."""

    if file.content_type not in {"application/pdf", "application/x-pdf", "binary/octet-stream"}:
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported.")

    temp_dir = Path(tempfile.mkdtemp(prefix="rag-pipeline-"))
    pdf_path = temp_dir / file.filename

    try:
        with pdf_path.open("wb") as fh:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)

        output_dir = temp_dir / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            process_pdf_for_rag(str(pdf_path), str(output_dir), doc_id=doc_id)
        except Exception as exc:  # pragma: no cover - surfaced to API consumer
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}") from exc

        archive_base = temp_dir / "rag_artifacts"
        archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=output_dir)

        background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)

        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=f"{doc_id}-rag-artifacts.zip",
        )
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


@app.get("/health")
async def health() -> JSONResponse:
    """Simple readiness probe for hosting platforms."""

    return JSONResponse({"status": "ok"})
