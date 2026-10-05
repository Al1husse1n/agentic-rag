from langchain_core.documents import Document

from main.schemas import RerankedResult


def format_reranked_results(
    reranked_results: list[RerankedResult],
) -> str:
    sections = []

    for i, result in enumerate(reranked_results, start=1):
        question = result["question"]
        documents: list[Document] = result["documents"]

        section = f"Question {i}:\n{question}\n\nEvidence:\n"

        for j, doc in enumerate(documents, start=1):
            section += (
                f"[Document {j}]\n"
                f"{doc.page_content}\n"
            )

        sections.append(section)

    return "\n\n".join(sections)