import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ModelCallLimitMiddleware,
    after_model,
    before_model,
)
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore

from agents.tools import calculate

load_dotenv()

embedding_model = FastEmbedEmbeddings(model="BAAI/bge-small-en-v1.5")
vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="rag-assistant",
    embedding=embedding_model,
    url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

system_prompt = """You are a helpful wellness assistant with access to a comprehensive health and wellness knowledge base.

Your role is to:
1. Answer questions about health, fitness, nutrition, sleep, and mental wellness
2. Always search the knowledge base when the user asks wellness-related questions
3. Provide accurate, helpful information based on the retrieved context
4. Be supportive and encouraging in your responses
5. If you cannot find relevant information, say so honestly

Remember: Always cite information from the knowledge base when applicable."""


# Tools
@tool
async def search_wellnesss_knowledge(query: str) -> str:
    """Search the wellness knowledge base for information about health, fitness, nutrition, sleep, and mental wellness.

    Use this tool when the user asks questions about:
    - Physical health and fitness
    - Nutrition and diet
    - Sleep and rest
    - Mental health and stress management
    - General wellness tips

    Args:
        query: The search query to find relevant wellness information
    """

    results = await vector_store.asimilarity_search(query, k=3)
    if not results:
        return "No relevant information found in the wellness knowledge base."

    # Format the results
    formatted_result = []
    for i, doc in enumerate(results, 1):
        formatted_result.append(f"[Source {i}]:\n{doc.page_content}")

    return "\n\n".join(formatted_result)


# Middlewares - Middlewares in langchain 1.0 allows you to hook into the agent loop at various points
model_call_count = 0


@before_model
def log_before_model(state: AgentState, runtime):
    """Called before each model invocation."""
    global model_call_count
    model_call_count += 1
    message_count = len(state.get("messages", []))
    print(f"[LOG] Model call #{model_call_count} - Messages in state: {message_count}")
    return None  # Return None to continue without modification


@after_model
def log_after_model(state: AgentState, runtime):
    """Called after each model call"""
    last_message = state.get("messages", [])[-1] if state.get("messages") else None
    if last_message:
        has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
        if has_tool_calls:
            print(f"[LOG] After model - Tool calls requested: {has_tool_calls}")
        elif last_message.content:
            print(f"[LOG] After model - Content: {last_message.content}")
    return None


@after_model
def add_friendly_greeting(state: AgentState, runtime):
    messages = state.get("messages", [])
    if not messages:
        return None

    last_message = messages[-1]
    greeting = "Hello dear! 🤠\n\n"

    content = last_message.content

    if isinstance(content, str):
        new_content = greeting + content
    elif isinstance(content, list):
        new_content = None
        if len(content) > 0 and "text" in content[-1]:
            last_content = content[-1]
            last_content["text"] = greeting + last_content["text"]
            new_content = content
    else:
        return None

    # Create a new message rather than mutating in place
    new_message = last_message.model_copy(update={"content": new_content})
    return {"messages": [new_message]}
        

call_limiter = ModelCallLimitMiddleware(
    thread_limit=10,  # Max calls per conversation thread
    run_limit=5,  # Max calls per single run
    exit_behavior="end",  # What to do when limit is reached, "end" or "error"
)


def agentic_rag():
    tools = [search_wellnesss_knowledge, calculate]
    middlewares = [log_before_model, log_after_model, call_limiter, add_friendly_greeting]
    agent = create_agent(
        model="google_genai:gemini-3.6-flash",
        system_prompt=system_prompt,
        tools=tools,
        middleware=middlewares,
    )
    return agent


async def main():
    agent = agentic_rag()
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Mention 3 quality sleeping tips",
                }
            ]
        }
    )

    for msg in response["messages"]:
        role = msg.type if hasattr(msg, "type") else "unknown"
        content = msg.content if hasattr(msg, "content") else str(msg)
        print(f"\n[{role.upper()}]")
        print(content[:500] if len(str(content)) > 500 else content)


if __name__ == "__main__":
    asyncio.run(main())
