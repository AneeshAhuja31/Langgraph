from dotenv import load_dotenv

from langgraph.graph import END, StateGraph
from graph.consts import RETRIEVE,GRADE_DOCUMENTS,GENERATE,WEBSEARCH
from graph.nodes import generate,grade_documents,retrieve,web_search
from graph.state import GraphState
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
load_dotenv()

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "DECISION: NOT ALL DOCUMENTS ARE NOT RELEVVANT TO QUESTION---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE
def grade_generation_grounded_in_documents_and_question(state:GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents":documents,"generation":generation})
    is_grounded = score.binary_score
    if is_grounded:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION")
        score = answer_grader.invoke({"question":question,"generation":generation})
        addresses_question = score.binary_score
        if addresses_question:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESSES QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not supported"
    
graph = StateGraph(GraphState)

graph.add_node(RETRIEVE,retrieve)
graph.add_node(GRADE_DOCUMENTS,grade_documents)
graph.add_node(GENERATE,generate)
graph.add_node(WEBSEARCH,web_search)

graph.set_entry_point(RETRIEVE)
graph.add_edge(RETRIEVE,GRADE_DOCUMENTS)
graph.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH:WEBSEARCH,
        GENERATE:GENERATE
    }
)
graph.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported":GENERATE,
        "useful":END,
        "not useful":WEBSEARCH
    }
)
graph.add_edge(WEBSEARCH,GENERATE)
graph.add_edge(GENERATE,END)

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path="self_rag_graph.png")