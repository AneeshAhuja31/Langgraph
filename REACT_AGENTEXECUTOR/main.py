from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState,StateGraph
from nodes import run_agent_reasoning,tool_node

load_dotenv()

AGENT_REASON = "agent_reason" 
ACT = "act"
LAST = -1
