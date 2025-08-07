from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@tool
def triple(num:float):
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1),triple]

llm = ChatGroq(model="Llama3-8b-8192",api_key=GROQ_API_KEY).bind_tools(tools)