from dotenv import load_dotenv
load_dotenv()
import os

from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph import END,MessageGraph
from chains import GEMINI_API_KEY
"""
from langgraph.graph import StateGraph
class MessageGraph(StateGraph):

    # A StateGraph where every node 
    # - receives a list of messages as input
    # - returns one or more messages as output.

    def __init__(self):
        super().__init__(Annotated[list[AnyMessage],add_messages])

"""
from typing import Sequence,List

from chains import generate_chain,reflection_chain

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state:Sequence[BaseMessage]):
    return generate_chain.invoke({"messages":state})

def reflection_node(messages:Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflection_chain.invoke({"messages":messages})
    return [HumanMessage(content=res.content)]

builder = MessageGraph()
builder.add_node(GENERATE,generation_node)
builder.add_node(REFLECT,reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, {END:END,REFLECT:REFLECT})
builder.add_edge(REFLECT,GENERATE)

graph = builder.compile()
# print(graph.get_graph().draw_mermaid())

if __name__ == "__main__":
    print(GEMINI_API_KEY)
    inputs = HumanMessage(content="""
        Make this tweet better:"
        Just used @AstraDB for the first time and wow what an experience!
        Super smooth setup, built in vector search, and it just scales. 
        Combining structured + semantic search in one place?
        Yes please!🔥"
        """)
    response = graph.invoke(inputs)
    print(response)