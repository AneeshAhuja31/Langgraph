from typing import Any,Dict
from langchain.schema import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState
from dotenv import load_dotenv
load_dotenv()

web_search_tool = TavilySearch(max_results=3)

# def web_search(state:GraphState) -> Dict[str,Any]:
#     print("---WEB SEARCH---")
#     question = state["question"]
#     documents = state["documents"]
#     tavily_results = web_search_tool.invoke({"query":question})
#     joined_tavily_result = "\n".join(
#         [tavily_result["content"] for tavily_result in tavily_results]
#     )
#     web_results = Document(page_content=joined_tavily_result)
#     if documents is not None:
#         documents.append(web_results)
#     else:
#         documents = [web_results]
#     return {"documents":documents,"question":question}
# if __name__ == "__main__":
#     web_search(state={"question":"agent memory","documents":None})

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    try:
        tavily_results = web_search_tool.invoke({"query": question})
        
        if isinstance(tavily_results, dict):
            print("Tavily returned a dictionary")
            if "results" in tavily_results:
                results = tavily_results["results"]
                if isinstance(results, list):
                    content_list = []
                    for result in results:
                        if isinstance(result, dict):
                            content = result.get("content", str(result))
                            content_list.append(content)
                        else:
                            content_list.append(str(result))
                    joined_tavily_result = "\n".join(content_list)
                else:
                    joined_tavily_result = str(results)
            elif "content" in tavily_results:
                joined_tavily_result = str(tavily_results["content"])
            else:
                joined_tavily_result = str(tavily_results)
        else:
            print(f"Tavily returned unexpected type: {type(tavily_results)}")
            joined_tavily_result = str(tavily_results)
        
    except Exception as e:
        print(f"Error in web search: {e}")
        joined_tavily_result = f"Web search failed for query '{question}': {str(e)}"
    
    web_results = Document(page_content=joined_tavily_result)
    
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    return {"documents": documents, "question": question}
