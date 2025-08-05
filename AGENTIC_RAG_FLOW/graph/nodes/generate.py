from typing import Any, Dict
from AGENTIC_RAG_FLOW.graph.chains.generation import generation_chain
from AGENTIC_RAG_FLOW.graph.state import GraphState

def generate(state: GraphState) -> Dict[str,Any]:
    print("--GENERATE--")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context":documents,"question":question})
    return {"documents":documents,"question":question,"generation":generation}