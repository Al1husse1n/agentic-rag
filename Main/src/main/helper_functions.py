from langchain_core.documents import Document

from main.schemas import RerankedResult, RetrievedResult


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


def compress_retrieved_results(retrieved_results: list[RetrievedResult]) -> dict[str, list[str]]:
    return {
        result["question"]: [doc.page_content for doc in result["documents"]]
        for result in retrieved_results
    }


def compress_reranked_results(reranked_results: list[RerankedResult]) -> dict[str, list[str]]:
    return {
        result["question"]: [doc.page_content for doc in result["documents"]]
        for result in reranked_results
    }

