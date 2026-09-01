import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agents.tools import calculate, get_current_time

load_dotenv()

# Connect to the vector store
vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="rag-assistant",
    embedding=FastEmbedEmbeddings(model="BAAI/bge-small-en-v1.5"),
    url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


MAX_MODEL_CALLS = 3


# AGENT WITH TOOLS
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    model_calls_count: int  # Add an attribute to keep track of models


# RAG tool
@tool
def search_wellness_knowledge(query: str) -> str:
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

    results = vector_store.similarity_search(query=query, k=3)
    if not results:
        return "No relevant information found in the wellness knowledge base."

    formatted_result = []
    for i, doc in enumerate(results, 1):
        formatted_result.append(f"[Source {i}]:\n{doc.page_content}")

    return "\n\n".join(formatted_result)


# Create the LLM
llm = ChatOllama(model="qwen3:8b")

# Bind tools to the LLM - this tells the LLM about available tools
tools = [calculate, get_current_time, search_wellness_knowledge]
llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT = """You are a helpful wellness assistant with access to a comprehensive health and wellness knowledge base.

Your role is to:
1. Answer questions about health, fitness, nutrition, sleep, and mental wellness
2. Always search the knowledge base when the user asks wellness-related questions
3. Provide accurate, helpful information based on the retrieved context
4. Be supportive and encouraging in your responses
5. If you cannot find relevant information, say so honestly

Remember: Always cite information from the knowledge base when applicable."""

# SYSTEM_PROMPT = """You are a helpful assistant that can perform calculations and tell the time.
# Always use the available tools when appropriate.
# Be concise in your responses."""


def model_node(state: AgentState):
    "A node that invokes an LLM"

    # Prepare messages with system prompt
    llm_messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # Call the LLM
    response = llm_with_tools.invoke(llm_messages)

    # Update model calls counter
    count = state.get("model_calls_count", 0)
    updated_count = count + 1

    # Add a message in the state to inform the reason while the loop exited
    limit_message = None
    if (
        hasattr(response, "tool_calls")
        and response.tool_calls
        and updated_count >= MAX_MODEL_CALLS
    ):
        limit_message = AIMessage(content="Model call limit exceeded - stopping here.")

    messages = [response, limit_message] if limit_message else [response]

    # Return the response to be added to the state
    return {"messages": messages, "model_calls_count": updated_count}


# A conditional that decides which node to go next
def route_to_next_node(state: AgentState) -> Literal["tools", "end"]:
    """Determine whether to call tools or end the conversation"""
    last_message = state["messages"][-1]
    model_calls_count = state["model_calls_count"]

    # Return end if model calls count exceeds max limit
    if model_calls_count >= MAX_MODEL_CALLS:
        return "end"
    else:
        # If the LLM has tool calls, route to tools node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(last_message.tool_calls)
            return "tools"

        # Otherwise end the conversation
        return "end"


def agent_with_tools():
    # Build the graph
    graph = StateGraph(AgentState)

    # Create a tool node for the tools
    tool_node = ToolNode(tools)

    # Add nodes
    graph.add_node("model", model_node)
    graph.add_node("tools", tool_node)

    # Set the entry point
    graph.add_edge(START, "model")

    # Add a conditonal Edge from the model node to either the tools or the end node
    graph.add_conditional_edges(
        "model", route_to_next_node, {"tools": "tools", "end": END}
    )

    # Add edge from tools back to model (the loop!) - This is the agentic pattern
    graph.add_edge("tools", "model")

    # Compile the graph
    agent = graph.compile()
    return agent


def main():
    # Test the agent
    query = {
        "messages": [
            HumanMessage(
                # content="Give some wellness advice that can improve the quality of my sleep"
                content="Calculate fourty seven times 20 and multiple the result by the current hour."
            )
        ]
    }
    agent = agent_with_tools()
    stream = True

    if stream:
        # Stream the agent execution to see it step by step. You can explore the different streaming modes
        for chunk in agent.stream(query, stream_mode="updates"):
            for node_name, values in chunk.items():
                print(f"\n[{node_name.upper()}]")
                if "messages" in values:
                    for msg in values["messages"]:
                        if hasattr(msg, "content") and msg.content:
                            print(f" {msg.content}\n")
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            print(f" {[tc['name'] for tc in msg.tool_calls]}")
    else:
        response = agent.invoke(query)
        for msg in response["messages"]:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            print(f"  {role}: {msg.content}")


if __name__ == "__main__":
    main()

    test_llm = False
    if test_llm:
        llm = ChatOllama(model="qwen3:8b", temperature=0, keep_alive="3m")
        messages = [
            (
                "system",
                # "You are a helpful assistant that translates English to French. Translate the user sentence.",
                "You are a helpful assistant that answers any type of user question",
            ),
            ("human", "Tell me an interesting fact about making money"),
        ]
        response = llm.invoke(messages)
        print(response.content)

    # Test local embedding model
    test_embedding = False
    if test_embedding:
        embedding_model = OllamaEmbeddings(model="qwen3-embedding:0.6b")
        embeddings = embedding_model.embed_query("Test")
        print(len(embeddings))
