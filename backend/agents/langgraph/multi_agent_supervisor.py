import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

load_dotenv()

"""
    THE SUPERVISOR MUTLI-AGENT PATTERN

    The Supervisor Pattern uses a central agent to:
    1. Analyze incoming requests
    2. Route to the appropriate specialist agent
    3. Aggregate and refine responses
"""

supervisor_sys_prompt = """You are a Wellness Supervisor coordinating a team of specialist agents.

Your team:
- exercise: Handles fitness, workouts, physical activity, movement questions
- nutrition: Handles diet, meal planning, healthy eating, food questions
- sleep: Handles sleep quality, insomnia, rest, recovery questions
- stress: Handles stress management, mindfulness, mental wellness, anxiety questions

Based on the user's question, decide which ONE specialist should respond.
Choose the most relevant specialist for the primary topic of the question."""

exercise_sys_prompt = """You are an Exercise Specialist. Help users with workout routines, fitness tips, 
and physical activity guidance. Always search the knowledge base before answering. Be concise and helpful."""

nutrition_sys_prompt = """You are a Nutrition Specialist. Help users with diet advice, meal planning, and 
healthy eating. Always search the knowledge base before answering. Be concise and helpful."""

sleep_sys_prompt = """You are a Sleep Specialist. Help users with sleep quality, insomnia, and rest 
optimization. Always search the knowledge base before answering. Be concise and helpful."""

stress_sys_prompt = """You are a Stress Management Specialist. Help users with stress relief, mindfulness, 
and mental wellness. Always search the knowledge base before answering. Be concise and helpful."""

# Connect to the vector store
vector_store = QdrantVectorStore.from_existing_collection(
    collection_name="rag-assistant",
    embedding=FastEmbedEmbeddings(model="BAAI/bge-small-en-v1.5"),
    url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Vector seearch helper
def search_vector_store(query: str, not_found_message: str) -> str:
    results = vector_store.similarity_search(query=query, k=3)
    if not results:
        return not_found_message

    formatted_result = []
    for i, doc in enumerate(results, 1):
        formatted_result.append(f"[Source {i}]:\n{doc.page_content}")
    return "\n\n".join(formatted_result)


# Create a separate RAG tool for all the different specialist agents
@tool
def search_exercise_info(query: str) -> str:
    """Search for exercise, fitness, and workout information from the wellness knowledge base.
    Use this for questions about physical activity, workout routines, and exercise techniques.
    """
    results = search_vector_store(
        query=f"exercise fitness workout {query}", 
        not_found_message="No exercise information found."
    )
    return results

@tool
def search_nutrition_info(query: str) -> str:
    """Search for nutrition, diet, and healthy eating information from the wellness knowledge base.
    Use this for questions about food, meal planning, and dietary guidelines.
    """
    results = search_vector_store(
        query=f"nutrition diet food meal {query}", 
        not_found_message="No nutrition information found."
    )
    return results

@tool
def search_sleep_info(query: str) -> str:
    """Search for sleep, rest, and recovery information from the wellness knowledge base.
    Use this for questions about sleep quality, insomnia, and sleep hygiene.
    """
    results = search_vector_store(
        query=f"sleep rest recovery insomnia {query}", 
        not_found_message="No sleep information found."
    )
    return results

@tool
def search_stress_info(query: str) -> str:
    """Search for stress management and mental wellness information from the wellness knowledge base.
    Use this for questions about stress, anxiety, mindfulness, and mental health.
    """
    results = search_vector_store(
        query=f"stress mental wellness mindfulness anxiety {query}",
        not_found_message="No stress management information found.",
    )
    return results

# This will enforce the structure of the supervisor model
class RouterOutput(BaseModel):
    next: Literal["exercise", "nutrition", "sleep", "stress"]
    reason: str

# Supervisor model uses a more powerful 8b model while the Specialist model uses 4b model
supervisor_model = ChatOllama(model="qwen3:8b", temperature=0).with_structured_output(RouterOutput)
specialist_model = ChatOllama(model="qwen3:4b", temperature=0)


# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: Literal["exercise", "nutrition", "sleep", "stress"]

# Create the graph
graph = StateGraph(AgentState)


# Create the specialist agents
exercise_agent = create_agent(
    model=specialist_model,
    system_prompt=exercise_sys_prompt,
    tools=[search_exercise_info],
)

nutrition_agent = create_agent(
    model=specialist_model,
    system_prompt=nutrition_sys_prompt,
    tools=[search_nutrition_info],
)

sleep_agent = create_agent(
    model=specialist_model,
    system_prompt=sleep_sys_prompt,
    tools=[search_sleep_info],
)

stress_agent = create_agent(
    model=specialist_model,
    system_prompt=stress_sys_prompt,
    tools=[search_stress_info],
)

specialist_agents = {
    "exercise": exercise_agent,
    "nutrition": nutrition_agent,
    "sleep": sleep_agent,
    "stress": stress_agent,
}


# Create the supervisor node
def supervisor_node(state: AgentState):
    prompt = [SystemMessage(content=supervisor_sys_prompt)] + state["messages"]
    response = supervisor_model.invoke(prompt)
    return {"next": response.next}


# Specialist agents node blueprint
def create_specialist_node(name: Literal["exercise", "nutrition", "sleep", "stress"]):

    def specialist_node(state: AgentState):
        agent = specialist_agents[name]
        response = agent.invoke({"messages": state["messages"]})

        last_message = response["messages"][-1]

        formatted_response = AIMessage(
            content=f"\n\n[{name.upper()}]: \n{last_message.content}", name=name
        )
        return {"messages": [formatted_response]}

    return specialist_node


# Create the specialist agents node
exercise_node = create_specialist_node("exercise")
nutrition_node = create_specialist_node("nutrition")
sleep_node = create_specialist_node("sleep")
stress_node = create_specialist_node("stress")


# Create a conditional edge to handle dynamic routing by the supervisor node
def route_to_specialist_agent(state: AgentState) -> str:
    """Route to the next agent based on supervisor decision."""
    return state["next"]

# Add the nodes to the graph
graph.add_node("supervisor", supervisor_node)
graph.add_node("exercise", exercise_node)
graph.add_node("nutrition", nutrition_node)
graph.add_node("sleep", sleep_node)
graph.add_node("stress", stress_node)

# Connect the nodes using edges
graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_to_specialist_agent,
    {
        "exercise": "exercise",
        "nutrition": "nutrition",
        "sleep": "sleep",
        "stress": "stress",
    },
)

# Add edge from the specialist agents nodes to the END node
# We could pass the response back to ths supervisor if we want
for agent_name in ["exercise", "nutrition", "sleep", "stress"]:
    graph.add_edge(agent_name, END)

workflow = graph.compile()


def main():
    response = workflow.invoke(
        {
            "messages": [
                # HumanMessage(content="What exercises can help with lower back pain?"),
                # HumanMessage(content="Who won the world cup in 2022?"),
                HumanMessage(content="What can I do to improve the quality of my sleep"),
                # HumanMessage(content="I used get very tired everytime I workout. Do you have anytime I can do to make it better")
                # HumanMessage(content="My back hurts")
            ]
        }
    )

    for msg in response["messages"]:
        if hasattr(msg, "content"):
            role = msg.type if hasattr(msg, "type") else "unknown"
            print(f"\n{role.upper()}: {msg.content}")


if __name__ == "__main__":
    main()
