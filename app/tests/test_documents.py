"""Document Management and Ingestion Test Suite.

Verifies:
1. Document upload (`POST /api/v1/documents/upload`):
   - Successfully streaming and persisting valid PDF, DOCX, and TXT files.
   - Enforcement of allowed file extensions (rejection of .exe, .png, etc.).
   - Rejection of 0-byte empty files.
   - Path traversal sanitization (preventing directory breakout via malicious filenames).
   - RBAC permission enforcement (requiring `documents:write`).
2. Paginated document listing (`GET /api/v1/documents/`):
   - Correct total count and pagination slices.
   - RBAC permission enforcement (requiring `documents:read`).
3. Single document retrieval (`GET /api/v1/documents/{document_id}`):
   - Retrieval of existing documents by UUID.
   - 404 handling on non-existent document IDs.
"""

import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.document import DocumentSource

# ---------------------------------------------------------------------------
# Upload Endpoint Tests (POST /api/v1/documents/upload)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_valid_pdf_success(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
    isolate_upload_directory: Path,
):
    """Verify that an instructor can upload a valid PDF and register metadata."""
    file_content = b"%PDF-1.4 Mock PDF file content for test."
    files = {"file": ("admissions_guide.pdf", file_content, "application/pdf")}
    data = {
        "category": "Admissions",
        "audience": "Prospective Families",
        "vector_backend": "chroma",
    }

    response = await client.post(
        "/api/v1/documents/upload",
        headers=instructor_user["headers"],
        files=files,
        data=data,
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == "admissions_guide.pdf"
    assert payload["category"] == "Admissions"
    assert payload["audience"] == "Prospective Families"
    assert payload["status"] == "pending"
    assert payload["uploaded_by_id"] == str(instructor_user["id"])
    assert "KB" in payload["file_size"] or "MB" in payload["file_size"]

    # Verify that a physical file was created on disk in the isolated directory
    stored_files = list(isolate_upload_directory.glob("*_admissions_guide.pdf"))
    assert len(stored_files) == 1
    assert stored_files[0].read_bytes() == file_content


@pytest.mark.asyncio
async def test_upload_valid_docx_and_txt(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
):
    """Verify that .docx and .txt formats are accepted."""
    for filename, content, mime in [
        ("curriculum.docx", b"PK\x03\x04Mock DOCX binary", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("faq.txt", b"Plain text FAQ questions and answers.", "text/plain"),
    ]:
        response = await client.post(
            "/api/v1/documents/upload",
            headers=instructor_user["headers"],
            files={"file": (filename, content, mime)},
            data={"category": "General", "audience": "All"},
        )
        assert response.status_code == 201
        assert response.json()["filename"] == filename


@pytest.mark.asyncio
async def test_upload_unsupported_extension_rejected(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
):
    """Verify that disallowed file types (.png, .exe) return 400 Bad Request."""
    response = await client.post(
        "/api/v1/documents/upload",
        headers=instructor_user["headers"],
        files={"file": ("malware.exe", b"MZExecutable binary", "application/octet-stream")},
        data={"category": "System", "audience": "All"},
    )

    assert response.status_code == 400
    assert "unsupported file type" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_empty_file_rejected(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
):
    """Verify that a zero-byte file is rejected with 400 Bad Request."""
    response = await client.post(
        "/api/v1/documents/upload",
        headers=instructor_user["headers"],
        files={"file": ("empty.pdf", b"", "application/pdf")},
        data={"category": "Test", "audience": "All"},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_directory_traversal_sanitized(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
    isolate_upload_directory: Path,
):
    """Verify directory traversal payloads in filenames are sanitized to base names."""
    malicious_filename = "../../../../escaped.pdf"
    response = await client.post(
        "/api/v1/documents/upload",
        headers=instructor_user["headers"],
        files={"file": (malicious_filename, b"%PDF-1.4 Data", "application/pdf")},
        data={"category": "Security", "audience": "All"},
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "escaped.pdf"

    # Ensure file was saved inside the test directory, not outside
    stored_files = list(isolate_upload_directory.glob("*_escaped.pdf"))
    assert len(stored_files) == 1


@pytest.mark.asyncio
async def test_upload_permission_denied_for_student(
    client: httpx.AsyncClient,
    student_user: dict[str, Any],
):
    """Verify that a student lacking `documents:write` permission is denied (403)."""
    response = await client.post(
        "/api/v1/documents/upload",
        headers=student_user["headers"],
        files={"file": ("unauthorized.pdf", b"%PDF-1.4 test", "application/pdf")},
        data={"category": "Cheat Sheet", "audience": "Students"},
    )

    assert response.status_code == 403
    assert "missing required permission" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_unauthenticated(client: httpx.AsyncClient):
    """Verify unauthenticated requests cannot upload documents (401)."""
    response = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("guest.pdf", b"%PDF-1.4 test", "application/pdf")},
        data={"category": "Public"},
    )

    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Listing Endpoint Tests (GET /api/v1/documents/)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_documents_pagination(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
    test_session: AsyncSession,
):
    """Verify that listing documents returns correct totals and respects pagination limits."""
    # Seed 3 documents in DB
    for i in range(3):
        doc = DocumentSource(
            id=uuid.uuid4(),
            filename=f"doc_{i}.pdf",
            file_path=f"/fake/path/doc_{i}.pdf",
            file_size="100.0 KB",
            chunk_count=5,
            category="Curriculum",
            audience="General Public",
            status="indexed",
            vector_backend="chroma",
            uploaded_by_id=instructor_user["id"],
        )
        test_session.add(doc)
    await test_session.commit()

    # Query first page of limit 2
    response = await client.get(
        "/api/v1/documents/?skip=0&limit=2",
        headers=instructor_user["headers"],
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["items"]) == 2

    # Query second page of skip 2, limit 2
    response_p2 = await client.get(
        "/api/v1/documents/?skip=2&limit=2",
        headers=instructor_user["headers"],
    )
    assert response_p2.status_code == 200
    assert len(response_p2.json()["items"]) == 1


@pytest.mark.asyncio
async def test_list_documents_accessible_by_student(
    client: httpx.AsyncClient,
    student_user: dict[str, Any],
):
    """Verify that users with `documents:read` (like students) can view the list."""
    response = await client.get(
        "/api/v1/documents/",
        headers=student_user["headers"],
    )

    assert response.status_code == 200
    assert "items" in response.json()
    assert "total" in response.json()


# ---------------------------------------------------------------------------
# Detail Endpoint Tests (GET /api/v1/documents/{document_id})
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_document_by_id_success(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
    test_session: AsyncSession,
):
    """Verify fetching an existing document by ID returns 200 and matches properties."""
    doc_id = uuid.uuid4()
    doc = DocumentSource(
        id=doc_id,
        filename="single_test.pdf",
        file_path="/fake/path/single_test.pdf",
        file_size="512.0 KB",
        chunk_count=12,
        category="Research",
        audience="Faculty",
        status="pending",
        vector_backend="chroma",
        uploaded_by_id=instructor_user["id"],
    )
    test_session.add(doc)
    await test_session.commit()

    response = await client.get(
        f"/api/v1/documents/{doc_id}",
        headers=instructor_user["headers"],
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(doc_id)
    assert data["filename"] == "single_test.pdf"
    assert data["chunk_count"] == 12


@pytest.mark.asyncio
async def test_get_document_by_id_not_found(
    client: httpx.AsyncClient,
    instructor_user: dict[str, Any],
):
    """Verify querying a non-existent document UUID returns 404 Not Found."""
    random_id = uuid.uuid4()
    response = await client.get(
        f"/api/v1/documents/{random_id}",
        headers=instructor_user["headers"],
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
