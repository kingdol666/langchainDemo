import os

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from os import getenv
from dotenv import load_dotenv

os.environ['OPENROUTER_API_KEY'] = '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'
os.environ['OPENROUTER_BASE_URL'] = 'https://openrouter.ai/api/v1'
load_dotenv()

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = ChatOpenAI(
  openai_api_key=getenv("OPENROUTER_API_KEY"),
  openai_api_base=getenv("OPENROUTER_BASE_URL"),
  model_name="deepseek/deepseek-chat-v3-0324:free>"
  # model_kwargs={
  #   "headers": {
  #     "HTTP-Referer": getenv("YOUR_SITE_URL"),
  #     "X-Title": getenv("YOUR_SITE_NAME"),
  #   }
  # },
)

# llm_chain = LLMChain(prompt=prompt, llm=llm)
#
# question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
#
# print(llm_chain.run(question))
#
# messages = [
#     SystemMessage("Translate the following from English into Italian"),
#     HumanMessage("hi!"),
# ]
#
# response = llm.invoke(messages)
# print(response)
#
# system_template = "Translate the following from English into {language}"
#
# prompt_template = ChatPromptTemplate.from_messages(
#     [("system", system_template), ("user", "{text}")]
# )
#
# prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
#
# print(prompt.to_messages())
# # response = llm.invoke(prompt)
# # print(response.content)
#
# # 流输出
# for token in llm.stream(prompt):
#     print(token.content, end="|")