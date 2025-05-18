# --- 确保在最顶部加载环境变量 ---
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

# --- 导入必要的库 ---
import requests
import json
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os
import re
import base64 # 用于Base64转换
from langchain.agents import AgentExecutor, create_react_agent # 或 create_tool_calling_agent
from langchain import hub
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage # 用于构建多模态消息
from langchain_community.tools.tavily_search import TavilySearchResults # <-- 新增导入
from rag_embedding import qa_chain
# --- 获取 API 密钥和 Base URL 从环境变量 ---
# 确保您的 .env 文件中设置了 MODELSCOPE_API_KEY 和 MODELSCOPE_BASE_URL
modelscope_api_key = os.getenv("MODELSCOPE_API_KEY")
modelscope_base_url = os.getenv("MODELSCOPE_BASE_URL")
tavily_api_key = os.getenv("TAVILY_API_KEY") # <-- 新增Tavily API Key获取

if not modelscope_api_key:
    raise ValueError("MODELSCOPE_API_KEY environment variable not set.")
if not modelscope_base_url:
    # ModelScope BASE_URL is often required for both text and multimodal models
    raise ValueError("MODELSCOPE_BASE_URL environment variable not set.")
if not tavily_api_key: # <-- 新增Tavily API Key检查
    raise ValueError("TAVILY_API_KEY environment variable not set. Please set it in your .env file.")


# --- 1. 定义并实例化多模态模型 (Qwen2.5-VL) ---
# 这个模型实例可以在 Tool 内部使用
multimodal_model_name = "Qwen/Qwen2.5-VL-72B-Instruct" # 确保这是 ModelScope 上正确的模型ID

# 实例化 ChatOpenAI for Multimodal
# 注意：ModelScope 的多模态 API 可能需要特定的 model_kwargs，请查阅文档
# 暂时不加 extra_body，如果报错再根据错误信息添加
chatLLM_multimodal = ChatOpenAI(
    openai_api_key=modelscope_api_key,
    openai_api_base=modelscope_base_url,
    model_name=multimodal_model_name,
    temperature=0.8 # 对多模态任务通常希望结果更客观准确
)

# --- Base64 转换函数 (如果需要处理本地图片) ---
# 如果你的 Agent 需要能够处理用户上传的本地图片，这个函数会很有用
def read_image_file(image_path: str) -> bytes:
    """
    读取图片文件并返回原始字节数据
    
    Args:
        image_path: 本地图片文件路径
        
    Returns:
        图片字节数据，如果失败则返回None
    """
    try:
        with open(image_path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to read image file: {e}")
        return None


def process_image(image_bytes: bytes, max_size: tuple = (768, 768)) -> Image.Image:
    """
    处理图片数据，包括打开、缩放和格式转换
    
    Args:
        image_bytes: 图片字节数据
        max_size: 缩放后的图片最大尺寸 (宽度, 高度)
        
    Returns:
        处理后的Pillow Image对象，如果失败则返回None
    """
    try:
        print(f"DEBUG: Original image file size: {len(image_bytes)} bytes")
        img = Image.open(BytesIO(image_bytes))
        print(f"DEBUG: Original image size (W, H): {img.size}")
        
        # 缩放图片
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        print(f"DEBUG: Resized image size (W, H): {img.size} (max_size={max_size})")
        
        # 处理格式转换
        if img.format == 'JPEG' and img.mode == 'RGBA':
            img = img.convert('RGB')
            
        return img
    except UnidentifiedImageError:
        print("ERROR: Could not identify image format")
        return None
    except Exception as e:
        print(f"ERROR: Failed to process image: {e}")
        return None


def encode_image_to_base64(img: Image.Image) -> str:
    """
    将Pillow Image对象编码为Base64 Data URL
    
    Args:
        img: Pillow Image对象
        
    Returns:
        Base64 Data URL字符串，如果失败则返回None
    """
    try:
        buffered = BytesIO()
        
        # 根据原始格式保存
        save_format = img.format if img.format in ['JPEG', 'PNG', 'GIF'] else 'PNG'
        mime_type = f'image/{save_format.lower()}'
        
        if save_format == 'JPEG':
            img.save(buffered, format="JPEG", quality=85)
        else:
            img.save(buffered, format=save_format)
            
        base64_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
        print(f"DEBUG: Final Base64 string length: {len(base64_encoded)}")
        
        return f"data:{mime_type};base64,{base64_encoded}"
    except Exception as e:
        print(f"ERROR: Failed to encode image: {e}")
        return None


def image_to_base64_data_url(image_path: str, max_size: tuple = (768, 768)) -> str:
    """
    将本地图片文件转换为Base64 Data URL
    
    Args:
        image_path: 本地图片文件路径
        max_size: 缩放后的图片最大尺寸 (宽度, 高度)
        
    Returns:
        Base64 Data URL字符串，如果失败则返回None
    """
    # 1. 读取图片文件
    image_bytes = read_image_file(image_path)
    if not image_bytes:
        return None
        
    # 2. 处理图片
    img = process_image(image_bytes, max_size)
    if not img:
        return None
        
    # 3. 编码为Base64
    return encode_image_to_base64(img)


# --- 新增：定义联网搜索 Tool --- 
search_tool = TavilySearchResults(max_results=3) # <-- 实例化搜索工具

# --- 9. 定义图片下载辅助函数 (从 Agent 输出中提取 URL并下载，与之前相同) ---
def extract_image_url(agent_output_string: str) -> str:
    """
    Extracts image URL from agent output string.
    
    Args:
        agent_output_string: The output string from the agent. Expected to contain the image URL returned by the tool.
                             Could be just the URL string, or a markdown link like "...[text](url)".
    
    Returns:
        Extracted image URL if found, otherwise empty string.
    """
    # First, try to extract URL from a markdown link if present
    match = re.search(r'\[.*?\]\((.*?)\)', agent_output_string)
    if match:
        return match.group(1)
    else:
        # If no markdown link, assume the output string itself is the URL
        return agent_output_string.strip()


def download_image(image_url: str) -> bytes:
    """
    Downloads image from given URL.
    
    Args:
        image_url: The URL of the image to download.
    
    Returns:
        Image content as bytes if successful, otherwise raises exception.
    """
    if not image_url or not (image_url.startswith('http://') or image_url.startswith('https://')):
        raise ValueError(f"Invalid image URL: {image_url}")
        
    print(f"Attempting to download image from URL: {image_url}")
    response = requests.get(image_url, stream=True, timeout=30)
    response.raise_for_status()
    return response.content


def save_image(image_content: bytes, save_directory: str = "images/", filename: str = None) -> str:
    """
    Saves image content to a local file.
    
    Args:
        image_content: The image content as bytes.
        save_directory: The directory where the image should be saved (default: current directory ".").
        filename: The filename to use. If None, will use default name.
    
    Returns:
        Full path to saved image if successful, otherwise raises exception.
    """
    # Get current file directory and combine with save_directory
    current_dir = os.path.dirname(__file__)
    full_save_dir = os.path.join(current_dir, save_directory)
    
    # Ensure the save directory exists
    os.makedirs(full_save_dir, exist_ok=True)
    
    # Use provided filename or default
    img_name = filename or "downloaded_image.png"
    full_save_path = os.path.join(full_save_dir, img_name)
    
    # Open and save the image
    image = Image.open(BytesIO(image_content))
    image.save(full_save_path, format='PNG')
    return full_save_path


def download_image_from_agent_output(agent_output_string: str, save_directory: str = "images/") -> bool:
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
    try:
        # Extract URL
        image_url = extract_image_url(agent_output_string)
        if not image_url:
            print(f"Error: Could not find a valid image URL in the agent output: {agent_output_string}")
            return False
            
        # Download image
        image_content = download_image(image_url)
        
        # Extract filename from URL if possible
        img_name = os.path.basename(image_url)
        if not img_name:
            img_name = None
            print("Warning: Could not extract filename from URL. Using fallback name.")
        
        # Save image
        full_save_path = save_image(image_content, save_directory, img_name)
        print(f"Successfully downloaded and saved image to {full_save_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return False
    except UnidentifiedImageError:
        print(f"Error: Could not identify image format from the downloaded content.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while processing or saving the image: {e}")
        return False


# --- 2. 定义文生图 Tool (保持不变) ---
def make_image_api_request(prompt: str) -> dict:
    """
    向ModelScope API发送图片生成请求
    
    Args:
        prompt: 图片描述文本
        
    Returns:
        包含API响应数据的字典，或抛出异常
    """
    url = 'https://api-inference.modelscope.cn/v1/images/generations'
    payload = {
        'model': 'AIkaiyuanfenxiangKK/chengxuyuan',
        'prompt': prompt
    }
    headers = {
        'Authorization': f'Bearer {modelscope_api_key}',
        'Content-Type': 'application/json'
    }
    
    response = requests.post(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers=headers,
        timeout=60
    )
    response.raise_for_status()
    return response.json()


def parse_image_api_response(response_data: dict) -> str:
    """
    解析ModelScope API的响应数据
    
    Args:
        response_data: API返回的JSON数据
        
    Returns:
        图片URL或错误信息
    """
    if 'images' in response_data and response_data['images']:
        image_url = response_data['images'][0].get('url')
        if image_url:
            return image_url
        return "Error: Image URL not found in the API response."
    elif 'error' in response_data:
        return f"Error from ModelScope API: {response_data['error'].get('message', 'Unknown error')}"
    return "Error: Unexpected response format from ModelScope API."

@tool
def generate_image_from_text(prompt: str) -> str:
    """
    Generates an image based on a text description (prompt) using the ModelScope text-to-image API.
    Input should be a string representing the image description.
    Returns the URL of the generated image if successful, or an error message.
    """
    try:
        response_data = make_image_api_request(prompt)
        image_url = parse_image_api_response(response_data)
        if image_url.startswith('http'):
            # print(f"Generated image URL: {image_url}, 已经成功生成出指定图片")
            download_image_from_agent_output(image_url)
        return f"Generated image URL: {image_url}, 已经成功生成出指定图片"
    except requests.exceptions.RequestException as e:
        return f"Error calling ModelScope API: {e}"
    except json.JSONDecodeError:
        return "Error: Failed to parse JSON response from ModelScope API."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# --- 3. 定义识图 Tool (使用 Qwen2.5-VL) ---
@tool
def describe_image_with_vl(input_string: str) -> str:
    """
    使用视觉语言模型(Qwen2.5-VL)回答关于图片的问题
    
    参数:
        input_string: 输入字符串，格式为 QUESTION: <问题> IMAGE: <图片URL或本地路径> \n
        注意严格按照：\nQUESTION: <问题> IMAGE: <图片URL或本地路径> \n这样的格式输入到input_string中\n

    返回:
        模型的描述或答案字符串
    """
    try:
        match = re.search(r'QUESTION:\s*(.*?)\s*IMAGE:\s*(.*)', input_string, re.DOTALL)
        if not match:
             return "Error: Input string is not formatted correctly. Expected 'QUESTION: <Your question> IMAGE: <Image URL or Local Path>'."

        question = match.group(1).strip()
        image_identifier = match.group(2).strip()

        if not question:
            return "Error: No question provided in the input string."
        if not image_identifier:
            return "Error: No image identifier provided in the input string."

        print(f"识图 Tool - Question: {question}, Image Identifier: {image_identifier[:50]}...")

        # 处理图片标识符(URL或本地路径)
        image_content_part = process_image_identifier(image_identifier)  # 处理图片标识符
        if isinstance(image_content_part, str):
            return image_content_part  # 返回错误信息

        # 构建多模态消息
        multimodal_message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                image_content_part
            ]
        )

        # 调用多模态模型
        print("Calling Qwen2.5-VL model...")
        response = chatLLM_multimodal.invoke([multimodal_message])
        print("Qwen2.5-VL response received.")
        return response.content

    except Exception as e:
        return f"An error occurred in the describe_image_with_vl tool: {e}"

@tool
def rag_agent(input_string: str) -> str:
    """
    基于RAG的问答系统

    参数:
        input_string: 输入需要在查询知识库中查询的问题，请你将输入的问题优化一下，方便文库智能体的查询

    返回:
        智能体的回答
    """
    try:
        response = qa_chain.invoke({"query": input_string})
        return response.get("result")
    except Exception as e:
        return f"An error occurred in the rag_agent tool: {e}"


def process_image_identifier(image_identifier: str) -> dict:
    """
    处理图片标识符(URL或本地路径)
    
    参数:
        image_identifier: 图片URL或本地路径
        
    返回:
        图片内容字典或错误字符串
    """
    # 检查是否是URL
    if image_identifier.lower().startswith('http://') or image_identifier.lower().startswith('https://'):
        print(f"\n{image_identifier}")
        print("Recognized image identifier as URL or Data URL.")
        return {"type": "image_url", "image_url": {"url": image_identifier}}
    
    # 处理本地路径
    print(f"Assuming image identifier is a local file path, attempting Base64 conversion: {image_identifier}")
    base64_data_url = image_to_base64_data_url(image_identifier)
    
    if not base64_data_url:
        return f"Error: Could not convert local image file '{image_identifier}' to Base64. Check file path or format."
    
    print("Successfully converted local path to Base64 Data URL.")
    return {"type": "image_url", "image_url": {"url": base64_data_url}}

# --- 4. 定义 Agent 需要使用的工具列表 (包含两个 Tool) ---
tools = [
    rag_agent,                 # <-- 新增 RAG 工具 (优先)
    generate_image_from_text, # 文生图 Tool
    describe_image_with_vl,   # 识图 Tool
    search_tool               # <-- 新增搜索工具
]

# --- 5. 定义 Agent 使用的 LLM (智能体的思考核心) ---
# 这个 LLM 负责理解用户指令并选择工具
# 确保 MODELSCOPE_BASE_URL 和 MODELSCOPE_API_KEY 已加载
# 选择一个强大的对话模型作为 Agent 的大脑
agent_llm = ChatOpenAI(
    openai_api_key=modelscope_api_key,
    openai_api_base=modelscope_base_url,
    model_name="Qwen/Qwen3-235B-A22B", # Replace with your Agent LLM model ID (e.g., Qwen/Qwen3-235B-A22B)
    temperature=0.9
    # model_kwargs={"extra_body": {"enable_thinking": False}} # Explicitly specify extra_body parameter
)

# --- 6. 获取 Agent Prompt ---
# 使用 ReAct 或 Tool Calling Prompt
# 优先使用 rag_agent 进行知识库查询
prompt = hub.pull("hwchase17/react")
prompt.template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do. If the question can be answered by querying the knowledge base (using 'rag_agent') or by searching the web (using 'search_tool'), you MUST prioritize using 'rag_agent'. Only use 'search_tool' if 'rag_agent' is not applicable or fails to provide a sufficient answer.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""" # Or hub.pull("hwchase17/tool-calling-agent")

# --- 7. 创建 Agent ---
# Create React Agent (more general) or Tool Calling Agent (if LLM supports function calling)
agent = create_react_agent(agent_llm, tools, prompt)
# If using Tool Calling (recommended for modern models like newer Qwen):
# from langchain.agents import create_tool_calling_agent
# agent = create_tool_calling_agent(agent_llm, tools, prompt)


# --- 8. 创建 Agent Executor ---
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True) # verbose=True helps debugging



# --- 10. 运行 Agent ---
print("Running agent...")

# 示例 1: 文生图请求
# user_query_generate = "请帮我生成一个可爱性感的妹子的全身照。"
# response_generate = agent_executor.invoke({"input": user_query_generate})
# print("\nAgent Response (Generate):")
# print(response_generate['output'])
# 尝试下载生成的图片


# print("\n" + "="*50 + "\n")

# 示例 2: 识图请求 (提供图片 URL)
# IMPORTANT: Your user input needs to include the image reference for the Agent to see it!
# The Agent LLM will read this string and decide if it matches the "describe_image_with_vl" tool's need.
# You might need to experiment with how your specific Agent LLM understands image references in text.
# Including the URL directly or using a marker is common.
# image_url_for_description = "https://modelscope.oss-cn-beijing.aliyuncs.com/demo/images/audrey_hepburn.jpg"
# user_query_describe_url = f"请描述这张图片里是谁？图片链接是：{image_url_for_description}"
# The Agent needs to parse this: "请描述这张图片里是谁？" is the question, "{image_url_for_description}" is the image identifier.
# The Agent LLM's reasoning steps (shown if verbose=True) will show how it tries to format the input for the tool.

# response_describe_url = agent_executor.invoke({"input": user_query_describe_url})


def useAgent(agent_executor: AgentExecutor, input: str) -> str:
    response = agent_executor.invoke({"input": input})
    res = "\nagent:" + response['output']
    return res

# print("\nAgent Response (Describe URL):")
# print(response_describe_url['output'])


# print("\n" + "="*50 + "\n")

# # 示例 3: 识图请求 (提供本地图片 Base64 数据 URL)
# # This requires you to first convert the local image to Base64 and include it in the user query string.
# local_image_path_for_description = "langchainDemo/images/2.png" # Replace with your local image path


# user_query_describe_base64 = f"这张图片里有什么？图片路径：{local_image_path_for_description}"
#     # Again, Agent needs to parse this.

# response_describe_base64 = agent_executor.invoke({"input": user_query_describe_base64})
# print("\nAgent Response (Describe Base64):")
# print(response_describe_base64['output'])
# 示例 4: 联网搜索请求
# user_query_search = "今天北京的天气怎么样？"
# response_search = agent_executor.invoke({"input": user_query_search})
# print("\nAgent Response (Search):")
# print(response_search['output'])