import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore

from agents.tools import calculate, fahrenheit_to_celcius, get_current_time, rand_int

load_dotenv()

embedding = FastEmbedEmbeddings(model="BAAI/bge-small-en-v1.5")
vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="rag-assistant",
    embedding=embedding,
    # path="./vector_store", # For local Qdrant store
    url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),  # For cloud hosted Qdrant
    api_key=os.getenv("QDRANT_API_KEY"),
)


def create_custom_agent():
    tools = [get_current_time, calculate, fahrenheit_to_celcius, rand_int]
    agent = create_agent(
        model="google_genai:gemini-3.6-flash",
        tools=tools,
        system_prompt="You are a helpful assistant that can perform calculations and tell the time. Always explain your reasoning.",
    )
    return agent


async def main():
    agent = create_custom_agent()
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the current time in GMT+1 timezone",
                }
            ]
        }
    )
    print("Full Agent Conversation:")
    print("=" * 50)
    for msg in response["messages"]:
        role = msg.type if hasattr(msg, "type") else "unknown"
        content = msg.content if hasattr(msg, "content") else str(msg)
        print(f"\n[{role.upper()}]")
        print(content[:500] if len(str(content)) > 500 else content)


if __name__ == "__main__":
    asyncio.run(main())
