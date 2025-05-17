from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from os import getenv
import base64 # 如果需要处理本地图片
import io # 如果需要处理本地图片

# 假设 ModelScope 提供了 Qwen2.5-VL 的 OpenAI 兼容 API 端点
# 请替换为 ModelScope 实际提供的 API 地址和模型名称
model_base_url = getenv("ModelScope_BASE_URL") # 例如 "https://dashscope.aliyuncs.com/compatible-mode/v1"
model_api_key = getenv("ModelScope_API_KEY")
# 请查阅 ModelScope 文档获取 Qwen2.5-VL 或类似模型的正确名称，可能是类似：
# "qwen-vl-max", "qwen-vl-beta", "Qwen/Qwen2.5-235B-A22B-VL" 等
multimodal_model_name = "Qwen/Qwen2.5-VL-72B-Instruct"

# 实例化 ChatOpenAI，指向 ModelScope 端点
# 注意：之前的 enable_thinking 参数是针对特定的非流式文本调用，
# 多模态模型可能不需要，如果需要，可以像之前那样通过 model_kwargs={"extra_body": {...}} 传入
chatLLM_multimodal = ChatOpenAI(
    openai_api_key=model_api_key,
    openai_api_base=model_base_url,
    model_name=multimodal_model_name,
    temperature=0.5, # 多模态视觉任务通常希望结果更确定
    max_tokens=1024, # 根据需要设置最大生成 tokens
    # model_kwargs={ ... } # 如果ModelScope的多模态API需要额外非标准参数，在这里添加
)

# --- 准备图片内容 ---

# 方法 1: 使用图片的公共 URL
image_url = "https://modelscope.oss-cn-beijing.aliyuncs.com/demo/images/audrey_hepburn.jpg"
image_content_part_url = {"type": "image_url", "image_url": {"url": image_url}}

# 方法 2: 使用本地图片文件，将其转换为 Base64 Data URL
def image_to_base64_data_url(image_path: str) -> str:
    """将本地图片文件转换为 Base64 Data URL"""
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
            # 获取图片类型（简单判断，实际应用可能需要更健壮的方式）
            if image_path.lower().endswith('.png'):
                mime_type = 'image/png'
            elif image_path.lower().endswith(('.jpg', '.jpeg')):
                mime_type = 'image/jpeg'
            else:
                mime_type = 'image/jpeg' # 默认或根据实际情况调整
            return f"data:{mime_type};base64,{base64_encoded}"
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

# 示例：假设你有一个本地图片文件 "path/to/your/image.jpg"
local_image_path = "langchainDemo\images/1.jpg" # 替换为你的图片路径
base64_data_url = image_to_base64_data_url(local_image_path)

if base64_data_url:
    image_content_part_base64 = {"type": "image_url", "image_url": {"url": base64_data_url}}
else:
    image_content_part_base64 = None
    print("Skipping base64 image part due to error.")


# --- 构造多模态 HumanMessage ---

# 使用图片 URL 和文本问题
multimodal_message_url = HumanMessage(
    content=[
        {"type": "text", "text": "请描述这张图片的内容。"}, # 你的文本问题
        image_content_part_url # 图片部分 (使用 URL)
    ]
)

# 或者使用 Base64 图片和文本问题
if image_content_part_base64:
    multimodal_message_base64 = HumanMessage(
        content=[
            {"type": "text", "text": "这张图片里有什么？"}, # 你的文本问题
            image_content_part_base64 # 图片部分 (使用 Base64)
        ]
    )
else:
    multimodal_message_base64 = None


# --- 调用模型 ---

# 使用 URL 图片进行调用
print("Calling model with image URL...")
try:
    response_url = chatLLM_multimodal.invoke([multimodal_message_url])
    print("Response (URL):")
    print(response_url.content)
except Exception as e:
    print(f"Error during URL call: {e}")


print("\n" + "="*30 + "\n")

# 使用 Base64 图片进行调用 (如果成功生成了 Base64 数据)
if multimodal_message_base64:
    print("Calling model with Base64 image...")
    try:
        response_base64 = chatLLM_multimodal.invoke([multimodal_message_base64])
        print("Response (Base64):")
        print(response_base64.content)
    except Exception as e:
        print(f"Error during Base64 call: {e}")