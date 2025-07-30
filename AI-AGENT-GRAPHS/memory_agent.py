import os
from typing import TypedDict,List,Union
from langchain_core.messages import HumanMessage,AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
class AgentState(TypedDict):
    messages:List[Union[HumanMessage,AIMessage]]

llm = ChatGroq(model="LLama3-8b-8192",api_key=GROQ_API_KEY)

def process(state:AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

converstational_history = []

user_input = input("Enter: ")
while user_input != "exit":
    converstational_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages":converstational_history})

    #print(result["messages"])
    converstational_history = result["messages"]

    user_input = input("Enter: ")