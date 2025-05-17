import requests
import json
from io import BytesIO
import os
from langchain.agents import AgentExecutor, create_react_agent # 或 create_tool_calling_agent (推荐较新模型)
from langchain import hub
from langchain.tools import tool # 使用 @tool 装饰器更方便
from langchain_openai import ChatOpenAI # 或者使用其他支持Tool Calling的ChatModel
from dotenv import load_dotenv # 如果使用 .env 文件管理 API 密钥
load_dotenv() # Load environment variables from .env file
from PIL import Image, UnidentifiedImageError # Import UnidentifiedImageError specifically
import re # Import the regular expression module
# 可选：加载环境变量，确保你的 ModelScope_API_KEY 在 .env 文件中
# load_dotenv()

# 获取 ModelScope API Key 从环境变量
modelscope_api_key = os.getenv("ModelScope_API_KEY")
modelscope_base_url = os.getenv("ModelScope_BASE_URL")
if not modelscope_api_key:
    raise ValueError("ModelScope_API_KEY environment variable not set.")

# 1. 将文生图逻辑封装成函数并使用 @tool 装饰器包装
# Tool 的名称会自动设为函数名，描述从 docstring 获取
@tool
def generate_image_from_text(prompt: str) -> str:
    """
    Generates an image based on a text description (prompt) using the ModelScope text-to-image API.
    Input should be a string representing the image description.
    Returns the URL of the generated image if successful, or an error message.
    """
    url = 'https://api-inference.modelscope.cn/v1/images/generations'
    payload = {
        # 注意：请根据 ModelScope 文档确认文生图模型的 Model-Id
        'model': 'AIkaiyuanfenxiangKK/chengxuyuan', # 示例 Model-Id，请替换为实际使用的文生图模型ID
        'prompt': prompt
    }
    headers = {
        'Authorization': f'Bearer {modelscope_api_key}',
        'Content-Type': 'application/json'
    }

    try:
        # 设置 timeout 防止长时间无响应
        response = requests.post(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers=headers,
            timeout=60 # 例如设置60秒超时
        )
        response.raise_for_status() # 检查HTTP响应状态码，如果不是2xx则抛出异常

        response_data = response.json()

        # 检查响应数据结构是否符合预期
        if 'images' in response_data and response_data['images']:
            # 提取第一个图片的 URL
            image_url = response_data['images'][0].get('url')
            if image_url:
                 # 可选：这里你可以选择是返回 URL，还是下载图片并保存，然后返回本地路径或成功的消息。
                 # 返回 URL 是 Agent 工具输出的常见方式，用户或后续流程可以自行处理 URL。
                 print(f"Generated image URL: {image_url}") # 打印以便调试查看
                 return image_url
            else:
                 return "Error: Image URL not found in the API response."
        elif 'error' in response_data:
             # 如果API返回了错误信息
             return f"Error from ModelScope API: {response_data['error'].get('message', 'Unknown error')}"
        else:
             return "Error: Unexpected response format from ModelScope API."

    except requests.exceptions.RequestException as e:
        return f"Error calling ModelScope API: {e}"
    except json.JSONDecodeError:
        return "Error: Failed to parse JSON response from ModelScope API."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# 2. 定义 Agent 需要使用的工具列表
tools = [generate_image_from_text]

# 3. 定义 Agent 使用的 LLM (智能体的思考核心)
# 您可以使用 LangChain 支持的任何 ChatModel，例如 OpenAI, Anthropic, 或 ModelScope 提供的兼容模型
# 这里继续使用 ChatOpenAI 作为示例，您可以根据您的需求替换为连接 ModelScope 的 ChatModel
# 确保 model_name 是一个适合作为 Agent 决策引擎的模型（通常是大型语言模型）
agent_llm = ChatOpenAI(
    openai_api_key=os.getenv("ModelScope_API_KEY"), # 假设 ModelScope 也用于 Agent 的 LLM
    openai_api_base=os.getenv("ModelScope_BASE_URL"), # 假设 ModelScope 也用于 Agent 的 LLM
    model_name="deepseek-ai/DeepSeek-V3-0324", # 替换为你的 Agent LLM 模型ID，例如 Qwen/Qwen3-235B-A22B
    temperature=0.8, # Agent 通常需要确定性强的决策
    # model_kwargs={"extra_body": {"enable_thinking": False}} # 如果 Agent LLM 需要此参数
)

# 4. 获取 Agent Prompt (ReAct 模式的 Prompt)
# LangChain Hub 提供了很多现成的 Agent Prompt
# 例如，使用 ReAct 模式的 Prompt
prompt = hub.pull("hwchase17/react")

# 5. 创建 Agent
# 使用 create_tool_calling_agent (推荐，需要模型支持 OpenAI Function/Tool Calling API)
# from langchain.agents import create_tool_calling_agent
# agent = create_tool_calling_agent(agent_llm, tools, prompt)

# 或者使用 create_react_agent (兼容性更好，但需要模型擅长ReAct模式)
agent = create_react_agent(agent_llm, tools, prompt)


# 6. 创建 Agent Executor (运行 Agent 的部分)
agent_executor_for_Image = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True 可以看到 Agent 的思考过程


# 7. 定义函数来提取 URL 并下载/保存图片
# --- Modified Algorithm to extract URL and download/save image using extracted name ---
def download_image_from_agent_output(agent_output_string: str, save_directory: str = ".") -> bool:
    """
    Extracts image URL from agent output string,
    downloads the image, and saves it to a local file in the specified directory
    using the filename extracted from the URL.

    Args:
        agent_output_string: The output string from the agent. Expected to contain the image URL returned by the tool.
                             Could be just the URL string, or a markdown link like "...[text](url)".
        save_directory: The directory where the image should be saved (default: current directory ".").

    Returns:
        True if the image was successfully downloaded and saved, False otherwise.
    """
    # First, try to extract URL from a markdown link if present
    match = re.search(r'\[.*?\]\((.*?)\)', agent_output_string)
    if match:
        image_url = match.group(1)
    else:
        # If no markdown link, assume the output string itself is the URL (less robust but might work)
        # You might want to add more checks here, like if it starts with "http"
        image_url = agent_output_string.strip() # Use strip to remove potential whitespace

    if not image_url or not (image_url.startswith('http://') or image_url.startswith('https://')):
        print(f"Error: Could not find a valid image URL in the agent output: {agent_output_string}")
        return False

    print(f"Attempting to download image from URL: {image_url}")

    try:
        # Extract filename from the URL
        # os.path.basename is robust and handles potential query parameters correctly
        img_name = os.path.basename(image_url)
        if not img_name: # Handle cases where URL might not have a simple filename part
            img_name = "downloaded_image.png" # Fallback name
            print(f"Warning: Could not extract filename from URL. Using fallback name: {img_name}")

        # Construct the full save path
        full_save_path = os.path.join(save_directory, img_name)

        # Ensure the save directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Download the image content
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()

        # Open the image from the downloaded content using Pillow
        image = Image.open(BytesIO(response.content))

        # Save the image (Pillow automatically handles format based on extension or can be specified)
        # Saving as PNG is generally safe and preserves quality/transparency
        image.save(full_save_path, format='PNG') # Explicitly save as PNG

        print(f"Successfully downloaded and saved image to {full_save_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {image_url}: {e}")
        return False
    except UnidentifiedImageError:
        print(f"Error: Could not identify image format from the downloaded content from {image_url}.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while processing or saving the image: {e}")
        return False

# --- 运行 Agent ---
print("Running agent...")
user_query = "请你给我生成一张粉色头发，金色眼睛，白皙皮肤的性感女孩的全身照图片，抖音风格。"
response = agent_executor_for_Image.invoke({"input": user_query})

print("\nAgent Response:")
agent_output = response['output']
print(agent_output)

# 如果 Agent 成功调用了文生图工具并返回了 URL，你可以在这里处理那个 URL，例如显示或者下载。
# 注意：Agent 返回的 output 是工具的返回值（在这里是 URL 字符串），或者 LLM 自身的文本回复。
# 你可能需要检查 output 是否是有效的 URL，或者 Agent 是否明确说明它已经生成了图片。
# 如果 output 是一个 URL 并且你希望自动下载：
# Call the function to download and save the image if the output format is correct
if download_image_from_agent_output(agent_output, save_directory="."):
    print("Image processing completed.")
else:
    print("Image processing failed.")