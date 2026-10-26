import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from langchain.agents import create_agent
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY") 

app = FastAPI(
    title="LangChain API",
    description="A simple API for the LangChain framework",
    version="0.1.0",
)


@tool
def search_web(query: str) -> str:
    """Search the web for the given query and return the results."""
    response = requests.get(
        "https://serpapi.com/search",
        params={
            "engine": "google",
            "q": query,
            "num": 5,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    snippets = []
    answer = data.get("answer_box") or {}
    if answer.get("answer"):
        snippets.append(str(answer["answer"]))
    elif answer.get("snippet"):
        snippets.append(str(answer["snippet"]))
    for item in data.get("organic_results", [])[:5]:
        title = item.get("title")
        snippet = item.get("snippet")
        if title and snippet:
            snippets.append(f"{title}: {snippet}")
        elif snippet:
            snippets.append(snippet)
    return "\n".join(snippets) or "No search results found."

chat_model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.0,
)


agent = create_agent(
    chat_model, 
    tools=[search_web],
    checkpointer=InMemorySaver(),
    )

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the today's weather in Tokyo and San Francisco?"}]},
    {"configurable": {"thread_id": "1"}},
)
print(result["messages"][-1].content)