import logging
import uuid
from pathlib import Path
from typing import Annotated
from uuid import UUID

import anyio
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from src.api.deps import CurrentUser, SessionDep, require_permission
from src.models.document import DocumentSource
from src.schemas.document import (
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Document Management"])


UPLOAD_DIR = Path("./data/pdfs").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  
CHUNK_SIZE_BYTES = 1024 * 1024  


def get_document_create_data(
    category: Annotated[
        str,
        Form(
            description="Document categorization (e.g., Admissions, Fees & Aid, Curriculum)",
            min_length=1,
            max_length=100,
        ),
    ],
    audience: Annotated[
        str,
        Form(
            description="Target audience group (e.g., Prospective Families, Current Students)",
            min_length=1,
            max_length=100,
        ),
    ] = "General Public",
    vector_backend: Annotated[
        str,
        Form(
            description="Target vector database backend (e.g., chroma, pinecone, pgvector)",
            min_length=1,
            max_length=50,
        ),
    ] = "chroma",
) -> DocumentCreate:

    return DocumentCreate(
        category=category.strip(),
        audience=audience.strip(),
        vector_backend=vector_backend.strip(),
    )



def format_file_size(bytes_count: int) -> str:
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f} KB"
    return f"{bytes_count / (1024 * 1024):.1f} MB"


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Register Document",
    description="Uploads a PDF, DOCX, or TXT file, persists it to local storage, and creates a tracking record in Neon.",
)
async def upload_document(
    current_user: CurrentUser,
    db: SessionDep,
    file: Annotated[UploadFile, File(description="Source document file (.pdf, .docx, or .txt)")],
    metadata: Annotated[DocumentCreate, Depends(get_document_create_data)],
    _: None = Depends(require_permission("documents:write")),
) -> DocumentSource:

    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must include a valid, non-empty filename.",
        )

    # Sanitize filename against directory traversal attacks (e.g., ../../etc/passwd)
    clean_filename = Path(file.filename).name
    file_extension = Path(clean_filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file_extension}'. Allowed formats: {allowed_list}.",
        )

    # Generate a collision-resistant unique storage filename on disk
    unique_storage_name = f"{uuid.uuid4().hex}_{clean_filename}"
    file_location = UPLOAD_DIR / unique_storage_name

    total_bytes = 0

    # Stream upload to disk asynchronously with size boundary checks
    try:
        async with await anyio.open_file(file_location, "wb") as buffer:
            while chunk := await file.read(CHUNK_SIZE_BYTES):
                total_bytes += len(chunk)

                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"File size exceeds maximum permitted limit of "
                            f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
                        ),
                    )

                await buffer.write(chunk)

    except HTTPException:
        # Clean up partial file on threshold breach
        if file_location.exists():
            file_location.unlink(missing_ok=True)
        raise

    except OSError as exc:
        if file_location.exists():
            file_location.unlink(missing_ok=True)
        logger.exception("Disk I/O error writing file %s", file_location)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist uploaded file to storage.",
        ) from exc

    finally:
        await file.close()

    # Enforce non-zero file size
    if total_bytes == 0:
        if file_location.exists():
            file_location.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes). Please upload a valid document.",
        )

    file_size_formatted = format_file_size(total_bytes)

    # Persist metadata record in Neon PostgreSQL using DocumentCreate fields
    db_document = DocumentSource(
        filename=clean_filename,
        file_path=str(file_location.resolve()),
        file_size=file_size_formatted,
        chunk_count=0,
        category=metadata.category,
        audience=metadata.audience,
        status="pending",
        vector_backend=metadata.vector_backend,
        uploaded_by_id=current_user.id,
    )

    try:
        db.add(db_document)
        await db.commit()
        await db.refresh(db_document)
    except SQLAlchemyError as exc:
        await db.rollback()
        # Ensure disk rollback: remove saved file if database write fails
        if file_location.exists():
            file_location.unlink(missing_ok=True)
        logger.exception("Database failure registering document %s", clean_filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist document metadata record to the database.",
        ) from exc

    logger.info(
        "Successfully registered document '%s' (ID: %s, Size: %s, Category: %s)",
        db_document.filename,
        db_document.id,
        db_document.file_size,
        db_document.category,
    )
    return db_document


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List Ingested Documents",
    description="Retrieves a paginated list of registered documents with total count metadata.",
)
async def list_documents(
    db: SessionDep,
    current_user: CurrentUser,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip for pagination")] = 0,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum number of records to return (1-100)")] = 50,
    _: None = Depends(require_permission("documents:read")),
) -> DocumentListResponse:

    try:
        count_stmt = select(func.count(DocumentSource.id))
        count_res = await db.execute(count_stmt)
        total = count_res.scalar_one()

        items_stmt = (
            select(DocumentSource)
            .order_by(DocumentSource.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        items_res = await db.execute(items_stmt)
        documents = list(items_res.scalars().all())

    except SQLAlchemyError as exc:
        logger.exception("Failed to query documents from database")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="A database error occurred while querying documents.",
        ) from exc

    return DocumentListResponse(
        total=total,
        items=[DocumentResponse.model_validate(doc) for doc in documents],
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get Document by ID",
    description="Fetches a single document's metadata and ingestion status by its unique UUID identifier.",
)
async def get_document_by_id(
    document_id: UUID,
    db: SessionDep,
    current_user: CurrentUser,
    _: None = Depends(require_permission("documents:read")),
) -> DocumentSource:
    
    try:
        stmt = select(DocumentSource).where(DocumentSource.id == document_id)
        result = await db.execute(stmt)
        document = result.scalars().first()
    except SQLAlchemyError as exc:
        logger.exception("Error retrieving document %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="A database error occurred while fetching document details.",
        ) from exc

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' was not found.",
        )

    return document
