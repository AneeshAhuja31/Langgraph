from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource"""

    datasource: Literal["vectorstore","websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web searh or to vectorstore."
    )

llm = ChatGroq(model="Llama3-8b-8192",api_key=GROQ_API_KEY)

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