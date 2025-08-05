from dotenv import load_dotenv

from langgraph.graph import END, StateGraph
from graph.consts import RETRIEVE,GRADE_DOCUMENTS,GENERATE,WEBSEARCH
from graph.nodes import generate,grade_documents,retrieve,web_search
from graph.state import GraphState

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
graph.add_edge(WEBSEARCH,GENERATE)
graph.add_edge(GENERATE,END)

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")