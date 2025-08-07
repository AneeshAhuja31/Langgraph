from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from react import llm,tools
load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions
"""

def run_agent_reasoning(state:MessagesState) -> MessagesState:
    """
    Run the agent reasoning node
    """ 
    response = llm.invoke([{"role":"system","content":SYSTEM_MESSAGE}])
    return {"messages":[response]}

tool_node = ToolNode(tools)