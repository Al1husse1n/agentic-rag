"""Tactical Golden Q&A Hotfix Cache Management.

Provides utilities for:
1. Parsing and maintaining markdown-based tactical hotfix overrides.
2. Fast-path LLM cache matching using Gemini Flash structured output.
3. Fallback fuzzy sequence matching for resilient offline evaluation.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_FILEPATH = Path("golden_qa.md")


class CacheMatchResult(BaseModel):
    matched: bool = Field(
        description="True if the student's question asks for the same information as one of the verified hotfixes. False if no hotfix addresses the question."
    )
    matched_index: int | None = Field(
        default=None,
        description="0-based index of the matching entry in the list (0 for Entry #0, 1 for Entry #1, etc.), or None if matched is False.",
    )
    reason: str | None = Field(
        default=None,
        description="Brief explanation of why it matched or why none matched.",
    )


def initialize_golden_qa_file(filepath: Path | None = None) -> None:
    """Ensures the golden_qa.md file exists with initial headers."""
    target_path = filepath or DEFAULT_FILEPATH
    if not target_path.exists():
        header = "# Golden Q&A Tactical Hotfixes\n\nApproved temporary overrides for course materials.\n\n---\n\n"
        target_path.write_text(header, encoding="utf-8")


def append_golden_qa_entry(
    question: str,
    answer: str,
    source_doc_id: str,
    instructor_id: str = "TA",
    filepath: Path | None = None,
) -> None:
    """Appends a new verified Q&A correction to the golden_qa.md file."""
    target_path = filepath or DEFAULT_FILEPATH
    initialize_golden_qa_file(target_path)

    entry = f"""### Q: {question.strip()}
* **source_doc_id:** {source_doc_id.strip()}
* **instructor_id:** {instructor_id.strip()}
* **answer:** {answer.strip()}

---
"""
    with open(target_path, "a", encoding="utf-8") as f:
        f.write(entry)


def parse_golden_qa_file(filepath: Path | None = None) -> list[dict[str, Any]]:
    """Parses the golden_qa.md file into structured entries."""
    target_path = filepath or DEFAULT_FILEPATH
    if not target_path.exists():
        return []

    content = target_path.read_text(encoding="utf-8")
    blocks = content.split("---")

    entries: list[dict[str, Any]] = []
    for block in blocks:
        q_match = re.search(r"### Q:\s*(.+)", block)
        doc_match = re.search(r"\*\s*\*\*source_doc_id:\*\*\s*(.+)", block)
        inst_match = re.search(r"\*\s*\*\*instructor_id:\*\*\s*(.+)", block)
        ans_match = re.search(r"\*\s*\*\*answer:\*\*\s*(.+)", block)

        if q_match and ans_match:
            entries.append({
                "question": q_match.group(1).strip(),
                "answer": ans_match.group(1).strip(),
                "source_doc_id": doc_match.group(1).strip() if doc_match else "unknown",
                "instructor_id": inst_match.group(1).strip() if inst_match else "TA",
            })

    return entries


def _get_cache_matcher_llm():
    """Lazily instantiate fast, zero-temperature Gemini Flash model for cache routing."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
    )


def _build_matcher_prompt(student_question: str, entries: list[dict[str, Any]]) -> str:
    """Format hotfix entries and student question into a concise cache routing prompt."""
    formatted_entries: list[str] = []
    for i, e in enumerate(entries):
        formatted_entries.append(
            f"Entry #{i}:\n"
            f"- Question: {e.get('question', '').strip()}\n"
            f"- Answer: {e.get('answer', '').strip()}\n"
            f"- Source Document: {e.get('source_doc_id', 'unknown')}"
        )
    entries_block = "\n\n".join(formatted_entries)

    return (
        "You are an authoritative course cache router evaluating student inquiries.\n"
        "Your task is to determine if the student's question is asking for the information provided "
        "by one of the verified instructor hotfixes listed below.\n\n"
        f"VERIFIED INSTRUCTOR HOTFIXES:\n{entries_block}\n\n"
        f"STUDENT QUESTION:\n{student_question.strip()}\n\n"
        "EVALUATION RULES:\n"
        "1. Match if the student's question asks for the same core information, even if "
        "phrased differently, using synonyms, or containing typos.\n"
        "2. Do NOT match if the student is asking for different information, even if it shares similar words "
        "(e.g., 'What is the graduation fee?' does NOT match 'When is graduation day?').\n"
        "3. If a hotfix answers the question, set matched=True and matched_index to its 0-based index.\n"
        "4. If no hotfix answers the question, set matched=False and matched_index=None."
    )


def _fuzzy_match(
    student_question: str,
    entries: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any] | None:
    """Fallback lexical sequence matcher using difflib."""
    best_match: dict[str, Any] | None = None
    highest_score = 0.0
    s_q_lower = student_question.strip().lower()

    for entry in entries:
        g_q_lower = str(entry["question"]).lower()
        score = SequenceMatcher(None, s_q_lower, g_q_lower).ratio()
        if score > highest_score and score >= threshold:
            highest_score = score
            best_match = {**entry, "match_score": round(score, 2)}

    return best_match


def check_golden_cache(
    student_question: str,
    threshold: float = 0.75,
    filepath: Path | None = None,
    use_llm: bool = True,
) -> dict[str, Any] | None:
    """Synchronous check of student question against cached Q&As using LLM or fuzzy matching."""
    entries = parse_golden_qa_file(filepath)
    if not entries:
        return None

    if use_llm:
        try:
            llm = _get_cache_matcher_llm()
            structured_llm = llm.with_structured_output(CacheMatchResult)
            prompt = _build_matcher_prompt(student_question, entries)
            result = structured_llm.invoke(prompt)

            if (
                isinstance(result, CacheMatchResult)
                and result.matched
                and result.matched_index is not None
                and 0 <= result.matched_index < len(entries)
            ):
                matched_entry = entries[result.matched_index]
                return {
                    **matched_entry,
                    "match_score": 1.0,
                    "match_reason": result.reason,
                }
            return None
        except Exception as e:
            logger.warning(
                "LLM golden cache matching failed (%s). Falling back to fuzzy sequence matcher.",
                e,
            )

    return _fuzzy_match(student_question, entries, threshold)


async def acheck_golden_cache(
    student_question: str,
    threshold: float = 0.75,
    filepath: Path | None = None,
    use_llm: bool = True,
) -> dict[str, Any] | None:
    """Asynchronous check of student question against cached Q&As using LLM or fuzzy matching."""
    entries = parse_golden_qa_file(filepath)
    if not entries:
        return None

    if use_llm:
        try:
            llm = _get_cache_matcher_llm()
            structured_llm = llm.with_structured_output(CacheMatchResult)
            prompt = _build_matcher_prompt(student_question, entries)
            result = await structured_llm.ainvoke(prompt)

            if (
                isinstance(result, CacheMatchResult)
                and result.matched
                and result.matched_index is not None
                and 0 <= result.matched_index < len(entries)
            ):
                matched_entry = entries[result.matched_index]
                return {
                    **matched_entry,
                    "match_score": 1.0,
                    "match_reason": result.reason,
                }
            return None
        except Exception as e:
            logger.warning(
                "Async LLM golden cache matching failed (%s). Falling back to fuzzy sequence matcher.",
                e,
            )

    return _fuzzy_match(student_question, entries, threshold)
