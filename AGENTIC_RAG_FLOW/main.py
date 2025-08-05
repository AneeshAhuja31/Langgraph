# import sys
# import os
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()
# from graph.graph import app
from graph.graph import app
if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question":"what is agent memory"}))