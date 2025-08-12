from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(api_key = GEMINI_API_KEY,model="gemini-2.5-flash")

class GradeHallucinations(BaseModel):
    """Binary score for hallunication present in generation answer."""
    binary_score: bool = Field(
        description="Answer is grounded in the facts, true or false (boolean, not string)"
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether a LLM generation is grounded in / supported by a set of retrieved facts. \n
        Give a binary score 'true' or 'false'. 'true' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader

