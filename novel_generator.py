from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic_core.core_schema import model_fields_schema
from typing_extensions import TypedDict, List  # Added List
from langgraph.graph import StateGraph, START, END
import os
from os import getenv
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import Literal

# Environment variables
os.environ["OPENROUTER_API_KEY"] = getenv(
    "OPENROUTER_API_KEY",
    "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",
)
os.environ["OPENROUTER_BASE_URL"] = getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

os.environ["ModelScope_BASE_URL"] = getenv(
    "ModelScope_BASE_URL", "https://api-inference.modelscope.cn/v1"
)

os.environ["ModelScope_API_KEY"] = getenv(
    "ModelScope_API_KEY", "5afeb288-c40f-4578-a404-167dfa126140"
)

os.environ["TongYi_API_KEY"] = getenv(
    "TongYi_API_KEY",
    "sk-264b27ce10364ffab10f14132b1f8dfe",
)
os.environ["TongYi_BASE_URL"] = getenv(
    "TongYi_BASE_URL", "http://dashscope.aliyuncs.com/compatible-mode/v1"
)

chatLLM = ChatOpenAI(
    openai_api_key=getenv("ModelScope_API_KEY"),
    openai_api_base=getenv("ModelScope_BASE_URL"),
    model_name="Qwen/Qwen3-235B-A22B",
    # other params...
    temperature=0.9,  # 您可以设置其他参数，例如温度
    # max_tokens=..., # 其他标准参数
    # *** 解决错误的关键在于添加 model_kwargs 参数 ***
    model_kwargs={
        # extra_body 字典中的内容会被添加到 API 请求的 body 中
        "extra_body": {"enable_thinking": True}
    },
)

# chatLLM = ChatOpenAI(
#     openai_api_key=getenv("TongYi_API_KEY"),
#     openai_api_base=getenv("TongYi_BASE_URL"),
#     model_name="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#     # other params...
# )

# # Model definitions
# model1 = ChatOpenAI(
#     temperature=0.9,
#     openai_api_key=getenv("OPENROUTER_API_KEY"),
#     openai_api_base=getenv("OPENROUTER_BASE_URL"),
#     model_name="deepseek/deepseek-chat-v3-0324:free",
# )

# model2 = ChatOpenAI(
#     temperature=0.7,
#     openai_api_key=getenv("OPENROUTER_API_KEY"),
#     openai_api_base=getenv("OPENROUTER_BASE_URL"),
#     model_name="qwen/qwen3-235b-a22b:free",
# )


class Judge(BaseModel):
    quality: Literal["good", "bad"] = Field(
        description="Decide if the chapter is good or bad.",
    )


judger = chatLLM.with_structured_output(Judge)


def create_structured_evaluator_prompt(text):
    return (
        "你是一个评估小说章节质量并提供结构化反馈的AI助手。\n"
        "你必须回复一个符合以下精确模式的有效JSON对象；\n"
        "请你根据大众的喜好程度，以及文章的整体内容和完整度等方面作为评判依据；\n"
        '{"quality": "good" 或 "bad"}\n\n'
        f"评估这个章节: {text}\n"
        "请记住，只返回JSON对象，不要包含其他任何内容。"
    )


# Define state type
class NovelWorkflowState(TypedDict):
    user_topic: str  # Initial topic from the user
    novel_premise: str  # Overall premise/idea for the novel, generated once
    previous_chapter_content: str  # Content of the previously generated chapter
    current_chapter_outline: str  # Outline for the chapter currently being generated
    current_chapter_full_content: (
        str  # Full content of the chapter currently being generated
    )
    all_generated_chapters: List[str]  # List to store content of all chapters
    current_chapter_number: int  # Counter for chapters, 1 up to max_chapters
    max_chapters: int  # Maximum number of chapters to generate
    decision: str  # Decision for the current chapter
    current_chapter_retry_count: int  # Counter for retries on the current chapter
    max_retries_per_chapter: int  # Maximum retries allowed for a single chapter


# Node 1: Generate novel premise (if first chapter) or chapter outline
def generate_outline_node(state: NovelWorkflowState) -> NovelWorkflowState:
    chapter_number = state["current_chapter_number"]
    if chapter_number == 1:
        # First chapter: Generate the overall novel premise
        messages = [
            SystemMessage(
                "你是一个创意小说构思专家。请根据用户提供的主题，生成一个引人入胜的、包含主要情节、核心人物和基本世界观的小说大纲。这个大纲将作为整部多章节小说的基础。\n"
                f"请你根据小说的总章节数（{state['max_chapters']}），决定大纲的深度和主题以及分别给出每个章节的简要大纲，要求每个章节都有一个主题。\n"
                "你将小说中出现的人物命名新颖一些，避免用太常见的名字。\n"
                "重要提示：在构思大纲时，请务必避免引入任何可能导向暴力、非法、仇恨言论或成人主题的情节元素。请专注于创意和故事结构的搭建，确保所有内容均在可接受的社会和道德规范之内。"
            ),
            HumanMessage(state["user_topic"]),
        ]
        premise_res = chatLLM.invoke(messages)
        print(f"第 {chapter_number} 章大纲：{premise_res.content}")
        # For the first chapter, the premise itself can serve as a high-level outline for chapter 1
        return {
            "novel_premise": premise_res.content,
            "current_chapter_outline": f"第一章：故事的开端，引入主要人物和背景，基于以下小说大纲展开：\n{premise_res.content}",
        }
    else:
        # Subsequent chapters: Generate outline for the current chapter based on premise and previous chapter
        messages = [
            SystemMessage(
                f"你是一个多章节小说的章节规划师。当前是第 {chapter_number} 章。"
                f"小说的主题大纲是：\n{state['novel_premise']}\n\n"
                f"上一章节的内容是：\n{state['previous_chapter_content']}\n\n"
                "请基于以上信息，为当前章节生成一个简洁但有创意的具体情节概要，确保故事的连贯性和逐步发展。\n"
                "重要提示：在规划章节概要时，请务必避免引入任何可能导向暴力、非法、仇恨言论或成人主题的情节元素。请专注于故事情节的连贯性和逐步发展，确保所有内容均在可接受的社会和道德规范之内。"
            ),
            HumanMessage(
                f"请为第 {chapter_number} 章生成情节概要。请直接输出情节概要，不要包含其他额外说明。"
            ),
        ]
        outline_res = chatLLM.invoke(messages)
        print(f"第 {chapter_number} 章大纲：{outline_res.content}")
        return {"current_chapter_outline": outline_res.content}


# Node 2: Judge chapter quality and decide whether to continue or regenerate
def judge_quality_node(state: NovelWorkflowState) -> NovelWorkflowState:
    chapter_number = state["current_chapter_number"]
    # Ensure current_chapter_full_content is available
    if not state.get("current_chapter_full_content"):
        # This case should ideally not happen if the graph is structured correctly
        # (write_chapter -> judge_quality)
        # but as a fallback, consider it not good or raise an error
        print(f"警告：在评估第 {chapter_number} 章质量时，章节内容未找到。")
        return {"decision": "not good"}

    messages = [
        SystemMessage(
            f"你是一个多章节小说的质量评估师。当前正在评估第 {chapter_number} 章的质量。\n"
            f"小说的主题大纲是：\n{state['novel_premise']}\n\n"
            f"上一章的内容是：\n{state['previous_chapter_content']}\n\n"
            f"本章的大纲是：\n{state['current_chapter_outline']}\n\n"
            "如果质量良好，请输出 'good'。如果质量不佳，需要重写，请输出 'bad'。"
            "请用评估全部用英文输出"
            "请不要加入任何分析，直接输出 'good' 或 'bad'"
        ),
        HumanMessage(
            f"请你对本章的生成内容：\n{state['current_chapter_full_content']}\n\n进行评估。"
        ),
    ]
    judge_prompt = create_structured_evaluator_prompt(
        state["current_chapter_full_content"]
    )
    try:
        judge_res = judger.invoke(judge_prompt)
        print(f"第 {chapter_number} 章：{state['current_chapter_full_content']}")
        print(f"第 {chapter_number} 章质量评估决策：{judge_res}")
        print(f"第 {chapter_number} 章质量评估结果：{judge_res.quality}")
        current_decision = judge_res.quality
        if current_decision == "good":
            return {"decision": current_decision, "current_chapter_retry_count": 0}
        else:
            updated_retry_count = state.get("current_chapter_retry_count", 0) + 1
            return {
                "decision": current_decision,
                "current_chapter_retry_count": updated_retry_count,
            }
    except Exception as e:
        print(f"结构化输出解析失败，尝试直接处理响应: {str(e)}")
        # Fallback to handle plain string responses
        response = chatLLM.invoke(messages)
        decision = response.content.strip().lower()
        # 确保只取判定词
        parsed_decision = "bad"  # 默认为 bad
        if "good" in decision:
            parsed_decision = "good"
        elif "bad" in decision:  # 明确检查 bad，避免其他词干扰
            parsed_decision = "bad"

        print(f"第 {chapter_number} 章质量评估决策（原始文本）: {decision}")
        print(f"第 {chapter_number} 章质量评估结果（解析后）: {parsed_decision}")
        if parsed_decision == "good":
            return {"decision": parsed_decision, "current_chapter_retry_count": 0}
        else:
            updated_retry_count = state.get("current_chapter_retry_count", 0) + 1
            return {
                "decision": parsed_decision,
                "current_chapter_retry_count": updated_retry_count,
            }


# Node 2: Expand outline into full chapter content
def write_chapter_node(state: NovelWorkflowState) -> NovelWorkflowState:
    chapter_number = state["current_chapter_number"]
    system_prompt = (
        f"你是一位才华横溢的小说家。当前正在创作第 {chapter_number} 章。\n"
        f"小说的整体大纲是：\n{state['novel_premise']}\n\n"
    )
    if state["previous_chapter_content"]:
        system_prompt += f"上一章回顾：\n{state['previous_chapter_content']}\n\n"

    system_prompt += (
        "请根据以下本章情节概要，进行详细的扩写，使其成为一个情节丰富、描写生动、情感饱满的完整章节。\n"
        "确保与小说整体大纲和前文章节（如果有）保持一致性和连贯性。请直接输出章节内容，不要包含其他额外说明，例如 '好的，这是第 X 章的内容：' 等。\n"
        "确保每一篇文章都有输出一个标题，如：第X章-《{标题}》\n"
        "确保每一章字数不超过四千字，且确保每次只写一章内容并且写完整，不能有遗漏。请你务必完整的输出给我。\n"
        "重要提示：在创作过程中，请务必避免生成任何可能被视为暴力、非法、仇恨言论或成人主题的内容。请专注于故事情节的推进和人物塑造，确保所有内容均在可接受的社会和道德规范之内。"
    )

    messages = [
        SystemMessage(system_prompt),
        HumanMessage(f"本章情节概要：\n{state['current_chapter_outline']}"),
    ]
    chapter_content_res = chatLLM.invoke(messages)
    return {"current_chapter_full_content": chapter_content_res.content}


# Node 3: Save chapter to file and update state for next iteration
def save_chapter_and_update_state_node(state: NovelWorkflowState) -> NovelWorkflowState:
    chapter_content = state["current_chapter_full_content"]
    chapter_number = state["current_chapter_number"]
    filename = f"novel_chapter_{chapter_number}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"章节 {chapter_number}\n\n")
        f.write(chapter_content)
    print(f"章节 {chapter_number} 已保存到 {filename}")

    # Ensure all_generated_chapters is initialized if not present
    all_chapters_so_far = state.get("all_generated_chapters", [])
    updated_all_chapters = all_chapters_so_far + [chapter_content]

    return {
        "previous_chapter_content": chapter_content,  # Current becomes previous for next iteration
        "all_generated_chapters": updated_all_chapters,
        "current_chapter_number": chapter_number
        + 1,  # Increment for the next potential chapter
    }


# Conditional edge: Decide whether to continue to the next chapter or end
def should_continue_writing(state: NovelWorkflowState) -> str:
    # current_chapter_number was already incremented in save_chapter_and_update_state_node
    # So, if it's now 11 (for 10 chapters), we stop.
    if state["current_chapter_number"] <= state["max_chapters"]:
        return "generate_outline"  # Continue to the next chapter's outline generation
    else:
        return END  # End the workflow


def route_decision(state: NovelWorkflowState) -> str:
    decision = state["decision"]
    current_retry_count = state.get("current_chapter_retry_count", 0)
    max_retries = state.get("max_retries_per_chapter", 3)  # Default to 3 max retries

    if decision == "good":
        return "save_and_update"
    elif decision == "bad":
        if current_retry_count < max_retries:
            print(
                f"章节 {state['current_chapter_number']} 质量不佳，重试次数 {current_retry_count}/{max_retries}。正在重新生成..."
            )
            return "write_chapter"
        else:
            print(
                f"章节 {state['current_chapter_number']} 达到最大重试次数 {max_retries}。将保存当前版本并继续。"
            )
            return "save_and_update"  # Force save and continue to avoid infinite loop
    else:
        # This path should ideally not be reached if 'decision' is always 'good' or 'bad'.
        # Raising an error helps in debugging unexpected states.
        raise ValueError(f"未知的评估决策 '{decision}' 在路由时遇到。")


# Build the workflow graph
def create_multichapter_novel_workflow():
    builder = StateGraph(NovelWorkflowState)

    builder.add_node("generate_outline", generate_outline_node)
    builder.add_node("write_chapter", write_chapter_node)
    builder.add_node("judge_quality", judge_quality_node)
    builder.add_node("save_and_update", save_chapter_and_update_state_node)

    builder.add_edge(START, "generate_outline")
    builder.add_edge("generate_outline", "write_chapter")
    builder.add_edge("write_chapter", "judge_quality")
    builder.add_conditional_edges(
        "judge_quality",
        route_decision,
        {"save_and_update": "save_and_update", "write_chapter": "write_chapter"},
    )
    builder.add_conditional_edges(
        "save_and_update",
        should_continue_writing,
        {
            "generate_outline": "generate_outline",  # Loop back to generate next chapter's outline
            END: END,
        },
    )
    return builder.compile()  # 增加递归限制


# Main function to generate the novel
def generate_novel_chapters(topic: str, num_chapters: int = 10):
    workflow = create_multichapter_novel_workflow()
    initial_state = NovelWorkflowState(
        user_topic=topic,
        novel_premise="",
        previous_chapter_content="",
        current_chapter_outline="",
        current_chapter_full_content="",
        all_generated_chapters=[],
        current_chapter_number=1,
        max_chapters=num_chapters,
        decision="",
        current_chapter_retry_count=0,
        max_retries_per_chapter=3,  # Set max retries per chapter
    )
    print(f"启动小说生成工作流，目标章节数: {num_chapters}")
    # The invoke might take a while for multiple chapters
    final_state = workflow.invoke(initial_state, {"recursion_limit": 100})
    print("小说生成工作流执行完毕。")
    return final_state.get("all_generated_chapters", [])


# Example usage
if __name__ == "__main__":
    novel_topic = "题目：《归零代码：当意识成为数据商品》。主题是：近未来数据朋克背景下，一场横跨虚拟与现实的记忆猎杀，揭开人类文明最黑暗的「意识永生」交易链，主人公名字是「归零」。"
    number_of_chapters_to_generate = 10  # For testing, generate fewer chapters initially. Change to 10 for full request.

    print(
        f"开始生成小说，主题：'{novel_topic}'，计划生成 {number_of_chapters_to_generate} 个章节。"
    )
    generated_chapters_list = generate_novel_chapters(
        novel_topic, num_chapters=number_of_chapters_to_generate
    )

    print(
        f"\n===== 小说生成完毕 ({len(generated_chapters_list)}/{number_of_chapters_to_generate} 章节) ====="
    )
    if generated_chapters_list:
        # The content is saved to files, here we can just confirm
        print(
            f"所有 {len(generated_chapters_list)} 个章节已分别保存到 novel_chapter_N.txt 文件中。"
        )
        # Optionally, print a snippet of each chapter if desired for quick review
        # for i, chapter_text in enumerate(generated_chapters_list):
        #     print(f"\n--- 章节 {i+1} (片段) ---")
        #     print(chapter_text[:200] + "..." if len(chapter_text) > 200 else chapter_text)
    else:
        print("未能生成任何章节。请检查日志或错误信息。")
