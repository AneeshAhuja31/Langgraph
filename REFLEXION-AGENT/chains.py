import datetime
import os
from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_core.output_parsers import JsonOutputParser,PydanticToolsParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
# from schemas import AnswerQuestion





from typing import List
from pydantic import BaseModel,Field

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    answer: str = Field(description="~250 word detailed answer to the question")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(description="1-3 search queries for researching improvements to address the critique of your current answer.")






groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=groq_api_key,model="llama3-8b-8192")

parser = JsonOutputParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion,Reflection])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current Time:{time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction = "Provide a detailed ~250 word answer"
)

first_responder = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion],tool_choice="AnswerQuestion")

if __name__ == "__main__":
    human_message = HumanMessage(
        content="""Write about AI-Powered SOC / autonomous soc problem domain,
        list startups that do that and raised capital."""
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion],tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages":[human_message]})
    print(res)