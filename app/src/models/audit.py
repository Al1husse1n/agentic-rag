# What it powers: The right-hand column (Unresolved Visitor Inquiries) showing flagged queries, semantic match percentages, context gaps, and action buttons like "Draft & Add to KB" or "Escalate".

"""
class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_logs"
    id: int | None = Field(default=None, primary_key=True)
    visitor_id: str  # e.g., "Visitor #802"
    subportal: str   # e.g., "Admissions Chat Widget"
    user_query: str  # "What are the financial aid options..."
    generated_answer: str
    confidence_score: float # e.g., 0.45 (45% Semantic Match)
    context_gap_reason: str # "Scholarship matrix does not specify..."
    status: str = Field(default="pending") # pending, escalated, resolved
    created_at: datetime
"""