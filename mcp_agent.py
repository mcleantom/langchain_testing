from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, END, START, StateGraph
from langgraph.prebuilt import create_react_agent


from langchain_openai import ChatOpenAI
import asyncio
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")


async def main():
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "123"}}
    async with MultiServerMCPClient(
            {
                "weather": {
                    "url": "http://localhost:8000/sse",
                    "transport": "sse"
                }
            }
    ) as client:
        agent = create_react_agent(llm, tools=client.get_tools(), checkpointer=memory)
        while True:
            user_input = input("User: ")
            if user_input in ["quit", "q", "exit"]:
                return
            async for event in agent.astream({"messages": [HumanMessage(content=user_input)]}, config=config, subgraphs=True):
                print(event)


if __name__ == "__main__":
    asyncio.run(main())
