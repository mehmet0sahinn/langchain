"""
ReAct Chat CLI - LangChain + Tavily demo
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

# 1. Load the Model
model = OpenAI()

# 2. Tavily Searching Tool
search = TavilySearchResults(max_results=2)
tools = [search]

# 3. Memory
memory = SqliteSaver.from_conn_string(":memory:")

# 4. ReACT - Chat Prompt Template 
prompt = hub.pull("hwchase17/react-chat")

# 5. Create an ReACT Agent
agent = create_react_agent(model, tools, prompt)

# 6. Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, checkpoint=memory)
config = {"configurable": {"thread_id": "pasha00"}}

# 7.main
if __name__ == '__main__':
    chat_history = []
    while True:
        user_input = input("> ")
        chat_history.append(f"Human: {user_input}")
        response = []
        for chunk in agent_executor.stream(
            {
                "input": user_input,
                "chat_history": "\n".join(chat_history),
            },
            config
        ):
            if 'text' in chunk:
                print(chunk['text'], end='')
                response.append(chunk['text'])
        chat_history.append(f"AI: {''.join(response)}")
        print("\n-------")
