import asyncio
import os
import random

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore

load_dotenv()

embedding = FastEmbedEmbeddings(model="BAAI/bge-small-en-v1.5")
vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="rag-assistant",
    embedding=embedding,
    # path="./vector_store", # For local Qdrant store
    url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),  # For cloud hosted Qdrant
    api_key=os.getenv("QDRANT_API_KEY"),
)


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for any math calculations.

    Args:
        expression: A mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')
    """
    try:
        # Using eval with restricted globals for safety
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_current_time():
    """Get the current date and time. Use this when the user asks about the current time or date."""
    from datetime import datetime

    return (
        f"The current date and time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


@tool
def fahrenheit_to_celcius(f: float) -> float:
    """Converts temperature from Fahrenheit to Celcius

    Args:
        f: Temperature in Fahrenheit (float)
    Returns:
        Temperature in Celcius (float)
    """
    return (f - 32.0) * 5.0 / 9.0


@tool
def rand_int(start: int, end: int) -> int:
    """Generate a random number within a given range.
    Args:
        start: The start position of the range
        end: The end position of the range
    Returns:
        A random number between
    """
    return random.randint(start, end)


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
