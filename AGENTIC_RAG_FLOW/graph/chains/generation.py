from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(api_key = GEMINI_API_KEY,model="gemini-2.5-flash")

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()