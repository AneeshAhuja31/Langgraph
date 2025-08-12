from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(api_key = GEMINI_API_KEY,model="gemini-2.5-flash")

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource"""

    datasource: Literal["vectorstore","websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web searh or to vectorstore."
    )


structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","{question}")
    ]
)

question_router = route_prompt | structured_llm_router