import re
from pathlib import Path
from difflib import SequenceMatcher

DEFAULT_FILEPATH = Path("golden_qa.md")

def initialize_golden_qa_file(filepath: Path = DEFAULT_FILEPATH):
    """Ensures the golden_qa.md file exists with initial headers."""
    if not filepath.exists():
        header = "# Golden Q&A Tactical Hotfixes\n\nApproved temporary overrides for course materials.\n\n---\n\n"
        filepath.write_text(header, encoding="utf-8")

def append_golden_qa_entry(
    question: str, 
    answer: str, 
    source_doc_id: str, 
    instructor_id: str = "TA", 
    filepath: Path = DEFAULT_FILEPATH
):
    """Appends a new verified Q&A correction to the golden_qa.md file."""
    initialize_golden_qa_file(filepath)
    
    entry = f"""### Q: {question.strip()}
* **source_doc_id:** {source_doc_id.strip()}
* **instructor_id:** {instructor_id.strip()}
* **answer:** {answer.strip()}

---
"""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(entry)

def parse_golden_qa_file(filepath: Path = DEFAULT_FILEPATH) -> list[dict]:
    """Parses the golden_qa.md file into structured entries."""
    if not filepath.exists():
        return []
    
    content = filepath.read_text(encoding="utf-8")
    blocks = content.split("---")
    
    entries = []
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

def check_golden_cache(student_question: str, threshold: float = 0.75, filepath: Path = DEFAULT_FILEPATH) -> dict | None:
    """Checks incoming student question against cached Q&As using fuzzy ratio."""
    entries = parse_golden_qa_file(filepath)
    best_match = None
    highest_score = 0.0

    s_q_lower = student_question.strip().lower()

    for entry in entries:
        g_q_lower = entry["question"].lower()
        # Uses built-in difflib sequence matching score (0.0 to 1.0)
        score = SequenceMatcher(None, s_q_lower, g_q_lower).ratio()
        
        if score > highest_score and score >= threshold:
            highest_score = score
            best_match = {**entry, "match_score": round(score, 2)}

    return best_match