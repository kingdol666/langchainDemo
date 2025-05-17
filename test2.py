import os
from typing_extensions import Literal, TypedDict
from dotenv import load_dotenv
from os import getenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from IPython.display import display, Image
from langchain_core.messages import HumanMessage, SystemMessage
# Load environment variables

# Environment variables
os.environ["OPENROUTER_API_KEY"] = getenv(
    "OPENROUTER_API_KEY",
    "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",
)
os.environ["OPENROUTER_BASE_URL"] = getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)


llm = ChatOpenAI(
  openai_api_key=getenv("OPENROUTER_API_KEY"),
  openai_api_base=getenv("OPENROUTER_BASE_URL"),
  model_name="anthropic/claude-3.7-sonnet"
  # model_kwargs={
  #   "headers": {
  #     "HTTP-Referer": getenv("YOUR_SITE_URL"),
  #     "X-Title": getenv("YOUR_SITE_NAME"),
  #   }
  # },
)

# Graph state
class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str


# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it.",
    )


# Augment the LLM with schema for structured output
evaluator = llm.with_structured_output(Feedback)

# Create a more explicit prompt template for the evaluator
def create_structured_evaluator_prompt(joke_text):
    return (
        "You are an AI assistant that evaluates jokes and provides structured feedback.\n"
        "You must respond with a valid JSON object that matches this exact schema:\n"
        "{\"grade\": \"funny\" or \"not funny\", \"feedback\": \"your feedback here\"}\n\n"
        f"Evaluate this joke: {joke_text}\n"
        "Remember to ONLY respond with the JSON object, nothing else."
    )


# Nodes
def llm_call_generator(state: State):
    """LLM generates a joke"""

    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}


def llm_call_evaluator(state: State):
    """LLM evaluates the joke"""

    try:
        # Use the structured prompt template
        structured_prompt = create_structured_evaluator_prompt(state['joke'])
        grade = evaluator.invoke(structured_prompt)
        return {"funny_or_not": grade.grade, "feedback": grade.feedback}
    except Exception as e:
        print(f"Error in joke evaluation: {str(e)}")
        # Provide default values if parsing fails
        return {
            "funny_or_not": "not funny", 
            "feedback": "The joke evaluation failed. Please try a different joke format."
        }

# Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
def route_joke(state: State):
    """Route back to joke generator or end based upon feedback from the evaluator"""

    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"


# Build workflow
optimizer_builder = StateGraph(State)

# Add the nodes
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# Add edges to connect nodes
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {  # Name returned by route_joke : Name of next node to visit
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

# Compile the workflow
optimizer_workflow = optimizer_builder.compile()

# Show the workflow
display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))

# Invoke
state = optimizer_workflow.invoke({"topic": "Cats"})
print(state["joke"])