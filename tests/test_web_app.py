import io
from pathlib import Path
import zipfile

from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_index_page_served() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "PDF → RAG Pipeline" in response.text


def test_process_endpoint_returns_zip(tmp_path) -> None:
    pdf_path = Path("sample_docs") / "sample_tax_excerpt.pdf"
    with pdf_path.open("rb") as fh:
        response = client.post(
            "/process",
            files={"file": (pdf_path.name, fh, "application/pdf")},
            data={"doc_id": "test-doc"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
    assert any(name.endswith("document.md") for name in names)
    assert any(name.endswith("chunks.jsonl") for name in names)
