import unittest
from pathlib import Path

from src.main.utils.golden_qa import(
    append_golden_qa_entry,
    parse_golden_qa_file,
    check_golden_cache,
)


class TestGoldenQAService(unittest.TestCase):
    def setUp(self):
        self.test_file = Path("test_golden_qa.md")
        if self.test_file.exists():
            self.test_file.unlink()

    def tearDown(self):
        if self.test_file.exists():
            self.test_file.unlink()

    def test_append_and_parse(self):
        append_golden_qa_entry(
            question="What is the add/drop deadline?",
            answer="Two weeks from the start of the semester.",
            source_doc_id="academic_calendar_v1.pdf",
            instructor_id="Dr. Alemayehu",
            filepath=self.test_file,
        )

        entries = parse_golden_qa_file(self.test_file)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["question"], "What is the add/drop deadline?")
        self.assertEqual(entries[0]["source_doc_id"], "academic_calendar_v1.pdf")

    def test_fuzzy_matching_hit(self):
        append_golden_qa_entry(
            question="How do I apply for late course registration?",
            answer="Submit Form A2 to the registrar's office with advisor signature.",
            source_doc_id="registration_v2.pdf",
            filepath=self.test_file,
        )

        # Slightly rephrased student question
        student_query = "How to apply late course registration?"
        match = check_golden_cache(student_query, threshold=0.70, filepath=self.test_file)

        # Assert match is not None to satisfy Pyright type checker
        self.assertIsNotNone(match)
        assert match is not None  # Type narrowing for Pyright

        self.assertIn("Submit Form A2", match["answer"])
        self.assertGreaterEqual(match["match_score"], 0.70)

    def test_fuzzy_matching_miss(self):
        append_golden_qa_entry(
            question="What is the passing grade for DBMS?",
            answer="A minimum grade of C is required.",
            source_doc_id="cs_curriculum.pdf",
            filepath=self.test_file,
        )

        # Unrelated query should miss the cache
        student_query = "Where is the campus cafeteria located?"
        match = check_golden_cache(student_query, threshold=0.75, filepath=self.test_file)

        self.assertIsNone(match)

if __name__ == "__main__":
    unittest.main()