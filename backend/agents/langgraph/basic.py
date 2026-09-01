from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# Define the state
class SimpleState(TypedDict):
    messages: Annotated[list, add_messages]


# Define a node - function that modify the state
def echo_node(state: SimpleState):
    """A simple node that echoes the last message"""
    last_message = state["messages"][-1]
    new_message = AIMessage(content=f"You said: {last_message.content}")
    return {"messages": [new_message]}


def simple_langraph_agent():
    # Build the graph
    graph = StateGraph(SimpleState)

    # Add the echo_node to the graph
    graph.add_node("echo", echo_node)

    # Add edge - START -> echo_node
    graph.add_edge(START, "echo")

    # Add edge - echo_node -> END
    graph.add_edge("echo", END)

    # Compile the graph
    agent = graph.compile()
    return agent


if __name__ == "__main__":
    agent = simple_langraph_agent()
    response = agent.invoke({
        "messages": {"role": "user", "content": "My name is Imran"}
    })

    for msg in response["messages"]:
        print(msg.content)