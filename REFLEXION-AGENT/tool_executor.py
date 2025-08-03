from dotenv import load_dotenv
load_dotenv(override=True)

from typing import List
from langgraph.prebuilt import ToolInvocation
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from schemas import AnswerQuestion,ReviseAnswer
from chains import parser_pydantic

def execute_tools(state:List[BaseMessage]) -> List[ToolMessage]:
    tool_invocation : AIMessage = state[-1]
    parsed_tool_calls = parser_pydantic.invoke(tool_invocation)
    ids = []
    tool_invocations = []
    for parsed_call in parsed_tool_calls:
        for query in parsed_call.search_queries:
            tool_invocations.append(ToolInvocation(
                tool="tavily_search_results_json",
                tool_input = query
            ))
            
if __name__ == "__main__":
    print("Tool Executor Enter")

    human_message = HumanMessage(
        content="""Write about AI-Powered SOC/autonomous soc problem domain,
        list startups that do that and raised capital"""
    )

    answer = AnswerQuestion(
        answer = "",
        missing = "",
        superfluous = "",
        search_queries = [
            "AI-powered SOC startups funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-powered SOC startups"
        ],
        id="call_KpYHichFFEmLitHFvFhKy1Ra"
    )

    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content="",
                tool_calls = [
                    {
                        "name": AnswerQuestion.__name__,
                        "args":answer.dict(),
                        "id":"call_KpYHichFFEmLitHFFvFhKy1Ra"
                    }
                ]
            )
        ]
    )
    print(raw_res)