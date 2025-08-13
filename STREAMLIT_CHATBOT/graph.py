from langgraph.graph import StateGraph, START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GEMINI_API_KEY)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages":[response]}

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.set_entry_point("chat_node")
graph.add_edge("chat_node",END)

app = graph.compile(checkpointer=checkpointer)
