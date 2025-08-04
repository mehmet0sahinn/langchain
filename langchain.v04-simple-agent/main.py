"""
Simple Agent - LangChain & LangGraph ReAct Chat CLI
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# 1. Load the Model
model = ChatOpenAI()

# 2. Tavily Searching Tool
search = TavilySearchResults(max_results=2)
tools = [search]

# 3. ReACT Agent Executor
agent_executor = create_react_agent(model, tools)
config = {"configurable": {"thread_id": "pasha00"}}

if __name__ == '__main__':
    while (user := input("> ")) != "exit":
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user)]},
            config,
        ):
            print(chunk)
            print("------------------")
