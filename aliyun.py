from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from os import getenv

os.environ["TongYi_API_KEY"] = getenv(
    "TongYi_API_KEY", "sk-264b27ce10364ffab10f14132b1f8dfe",
)
os.environ["TongYi_BASE_URL"] = getenv(
    "TongYi_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

chatLLM = ChatOpenAI(
    openai_api_key=getenv("TongYi_API_KEY"),
    openai_api_base=getenv("TongYi_BASE_URL"),
    model_name="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    # other params...
)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你是谁？"}]
response = chatLLM.invoke(messages)
print(response.json())