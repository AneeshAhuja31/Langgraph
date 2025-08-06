from dotenv import load_dotenv
load_dotenv()
from AGENTIC_RAG_FLOW.graph.chains.retriever_grader import GradeDocuments, retrieval_grader
from AGENTIC_RAG_FLOW.graph.chains.generation import generation_chain
from AGENTIC_RAG_FLOW.ingestion import retriever
from AGENTIC_RAG_FLOW.graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader

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

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context":docs,"question":question})
    print(generation)

def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context":docs,"question":question})
    res : GradeHallucinations = hallucination_grader.invoke(
        {"documents":docs,"generation":generation}
    )
    assert res.binary_score

def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context":docs,"question":question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents":docs,
            "generation":"In order to make pizza we need to first start with the dough"
        }
    )
    assert not res.binary_score
