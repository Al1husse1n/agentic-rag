import time

from langchain_core.documents import Document

from main.schemas import RerankedResult, RetrievedResult


def format_reranked_results(
    reranked_results: list[RerankedResult],
) -> str:
    start = time.time()
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

    result_str = "\n\n".join(sections)
    print(f"⏱ format_reranked_results took {time.time() - start:.4f}s")
    return result_str


def compress_retrieved_results(retrieved_results: list[RetrievedResult]) -> dict[str, list[str]]:
    start = time.time()
    compressed = {
        result["question"]: [doc.page_content for doc in result["documents"]]
        for result in retrieved_results
    }
    print(f"⏱ compress_retrieved_results took {time.time() - start:.4f}s")
    return compressed


def compress_reranked_results(reranked_results: list[RerankedResult]) -> dict[str, list[str]]:
    start = time.time()
    compressed = {
        result["question"]: [doc.page_content for doc in result["documents"]]
        for result in reranked_results
    }
    print(f"⏱ compress_reranked_results took {time.time() - start:.4f}s")
    return compressed

