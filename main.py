from BaseModel import llm as model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# messages = [
#     SystemMessage("Translate the following from English into Italian"),
#     HumanMessage("hi!"),
# ]

# res = model.invoke(messages)

# print(res)

# def get_weather(city: str) -> str:  
#     """Get weather for a given city."""
#     return f"It's always sunny in {city}!"

# agent = create_react_agent(
#     model=model,
#     tools=[get_weather],  
#     prompt="You are a helpful assistant"  
# )

# # Run the agent
# res = agent.invoke(
#     {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
# )
class TempState(TypedDict):
    memoryNovel: str
    newMemory: str
    Topic: str
    output: str

def node_1(state: TempState) -> TempState:
    res = model.invoke(state["Topic"])
    # Write to OverallState
    return {"newMemory": res["messages"][-1].content}

def node_2(state: TempState) -> TempState:
    # Read from OverallState, write to PrivateState
    res = model.invoke(state["newMemory"])
    return {"output": res["messages"][-1].content}


builder = StateGraph(TempState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

graph = builder.compile()
res = graph.invoke({"Topic":"武侠小说，穿越情节，描写的是主人公穿越时空的经历，通常会涉及到时空穿梭、时空穿梭门、时空裂缝等概念。"})

print(res["messages"][-1].content)