from typing import Annotated,Sequence,TypedDict
from dotenv import load_dotenv
import os
from langchain_core.messages import BaseMessage,ToolMessage,SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
"""
# Annotated - provides additional context without affecting the type itself. --------------------------------------------⤵️
# Sequence - To automatically handle the state updates for sequences such as by adding new messages to a chat history.    #
email = Annotated[str,"This has to be a valid email format!"] #⬅️--------------------------------------------------------⤴️

add_messages() #Reducer Function
# Rule that controls how updates from nodes are combined with the existing state
# Tells us how to merge new data into the current state.
# Without a reducer, updates would have replaced the existing value entirely.
"""
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages] # using Annotated, whenever a function updates Annotated, insteaad of replaced the messages list, it call add_messages to append the new message
"""
Annotated[Sequence[BaseMessage], add_messages] tells LangGraph:
-> The data is a list of messages.
-> When the state is updated, merge new messages instead of replacing.
"""
@tool
def add(a: int,b: int):
    """This is an addition function that adds 2 numbers together"""
    return a+b

@tool
def subtract(a: int,b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int,b: int):
    """Multiply function"""
    return a * b

tools = [add,subtract,multiply]

llm = ChatGroq(model="LLama3-8b-8192",api_key=GROQ_API_KEY).bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability, also you are illeterate in basic maths. You can only use the tools provided to do the maths")
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages":[response]}

def should_continue(state:AgentState):
    messages  = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: #check if the response from the llm is a tool call or not
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent",model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools",tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue":"tools",
        "end":END
    }
)

graph.add_edge("tools","our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [{"role": "user", "content": "Add 3+4 then multiply it by 10 and tell me a joke"}]}
print_stream(app.stream(inputs,stream_mode="values"))