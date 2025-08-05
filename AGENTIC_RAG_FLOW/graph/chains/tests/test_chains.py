from dotenv import load_dotenv
load_dotenv()
from AGENTIC_RAG_FLOW.graph.chains.retriever_grader import GradeDocuments, retrieval_grader
from AGENTIC_RAG_FLOW.ingestion import retriever

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question":question,"document":doc_txt}
    )

    assert res.binary_score == "yes"

def test_retreival_grader_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question":"how to make pizza","document":doc_txt}
    )

    assert res.binary_score == "no"