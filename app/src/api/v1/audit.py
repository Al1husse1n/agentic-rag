import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from src.api.deps import CurrentUser, SessionDep, require_permission
from src.models.audit import AuditLog
from src.schemas.audit import AuditLogListResponse, AuditLogResponse, AuditResolveRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit", tags=["HITL Audit & Inquiries"])


@router.get(
    "/",
    response_model=AuditLogListResponse,
    summary="List Unresolved / Audit Inquiries",
    description="Retrieves paginated HITL escalation logs for instructor review.",
)
async def list_audit_logs(
    db: SessionDep,
    current_user: CurrentUser,
    status_filter: Annotated[str | None, Query(alias="status", description="Filter by status (e.g., pending, resolved)")] = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    _: None = Depends(require_permission("audit:read")),
) -> AuditLogListResponse:
    """List audit logs with optional status filtering and pagination."""
    try:
        query = select(AuditLog)
        count_query = select(func.count(AuditLog.id))

        if status_filter:
            query = query.where(AuditLog.status == status_filter)
            count_query = count_query.where(AuditLog.status == status_filter)

        # Total count
        total_res = await db.execute(count_query)
        total = total_res.scalar_one()

        # Paginated items
        stmt = query.order_by(AuditLog.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(stmt)
        items = list(result.scalars().all())

    except SQLAlchemyError as exc:
        logger.exception("Database error while fetching audit logs")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit logs.",
        ) from exc

    return AuditLogListResponse(
        total=total,
        items=[AuditLogResponse.model_validate(item) for item in items],
    )


@router.patch(
    "/{audit_id}/resolve",
    response_model=AuditLogResponse,
    summary="Resolve Audit Inquiry",
    description="Allows an instructor to provide resolution notes and update an escalated inquiry's status.",
)
async def resolve_audit_log(
    audit_id: UUID,
    body: AuditResolveRequest,
    db: SessionDep,
    current_user: CurrentUser,
    _: None = Depends(require_permission("audit:write")),
) -> AuditLog:
    """Update an audit record with instructor notes, changing status to resolved or escalated."""
    try:
        stmt = select(AuditLog).where(AuditLog.id == audit_id)
        result = await db.execute(stmt)
        audit_item = result.scalars().first()

        if audit_item is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit log with ID '{audit_id}' not found.",
            )

        # Update record fields
        audit_item.instructor_notes = body.instructor_notes
        audit_item.status = body.status
        audit_item.reviewed_by_id = current_user.id

        await db.commit()
        await db.refresh(audit_item)

    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        await db.rollback()
        logger.exception("Database error resolving audit log %s", audit_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update audit log record.",
        ) from exc

    logger.info("Audit log %s resolved by instructor %s", audit_id, current_user.email_address)
    return audit_item