from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description= "Answer addresses the question, 'yes' or 'no'"
    )

llm = ChatGroq(model="Llama3-8b-8192",api_key=GROQ_API_KEY)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question.
Return a binary score: true (if the answer resolves the question) or false (if it does not).
Only return a boolean value, not a string like 'yes' or 'no'."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","User question: \n\n {question} \n\n LLM generation: {generation}")
    ]
)

answer_grader : RunnableSequence = answer_prompt | structured_llm_grader