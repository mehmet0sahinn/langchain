"""
A minimal LangChain + OpenAI chat CLI that keeps per‑session conversation history in memory.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.messages import HumanMessage


# 1. Load the Model
model = ChatOpenAI()

# 2. Chat Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 3. Parser
chain = prompt | model

# 4. History
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

config = {"configurable": {"session_id": "firstChat"}}
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

# 5. main
def main() -> None:
    print("Type 'exit' to quit.\n")
    while (user_input := input("> ")) != "exit":
        for chunk in with_message_history.stream(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        ):
            print(chunk.content, end="", flush=True)
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
