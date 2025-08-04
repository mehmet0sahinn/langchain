"""
Simple Agent - LangChain & LangGraph ReAct Chat CLI
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# 1. Load the Model
model = ChatOpenAI()

# 2. Tavily Searching Tool
search = TavilySearchResults(max_results=2)
tools = [search]

# 3. Bind Model and Tools
model_with_tools = model.bind_tools(tools)

# 4. Memory
memory = SqliteSaver.from_conn_string(":memory:")

# 5. ReACT Agent Executor
agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "pasha00"}}


if __name__ == '__main__':
    while True:
        user_input = input("> ")
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config
        ):
            print(chunk)
            print("------------------")
