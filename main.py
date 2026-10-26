import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI(
    title="LangChain API",
    description="A simple API for the LangChain framework",
    version="0.1.0",
)

chat_model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.0,
)

agent = create_agent(chat_model)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
)
print(result["messages"][-1].content)