from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState,StateGraph,START,END
from nodes import run_agent_reasoning,tool_node
import os
load_dotenv()

AGENT_REASON = "agent_reason" 
ACT = "act"
LAST = -1
output_dir = r"REACT_AGENTEXECUTOR"

def should_continue(state:MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        print("---END---")
        return END
    print("---ACT---")
    return ACT


flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON,run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT,tool_node)

flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {
        END:END,
        ACT:ACT
    }
)
flow.add_edge(ACT,AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path= "flow.png")

if __name__ == "__main__":
    res = app.invoke({"messages":[HumanMessage(content="WHat is the temperature in Tokyo? List it and then triple it")]})
    print(res["messages"][LAST].content)

