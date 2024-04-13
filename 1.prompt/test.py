import os  # 用于读取环境变量
import openai  # OpenAI 的 Python SDK，用于调用 API
from dotenv import load_dotenv, find_dotenv  # 用于读取 .env 文件

_ = load_dotenv(find_dotenv())  # 加载 .env 到环境变量

# 配置 OpenAI 服务
openai.api_key = os.getenv('OPENAI_API_KEY')  # 设置 OpenAI 的 key
openai.api_base = os.getenv('OPENAI_API_BASE')  # 指定代理地址

# 调用 OpenAI API
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "你的模型是基于GPT-3还是GPT-3.5"
        },
        {
            "role": "user",
            "content": "周末上课吗？"  # 问问题。可以改改试试
        },
    ],
)

print(response.choices[0].message["content"])
