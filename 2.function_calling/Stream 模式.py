from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import sqlite3
import json

_ = load_dotenv(find_dotenv())

client = OpenAI()

def print_json(data):
    """
    打印参数。如果参数是有结构的（如字典或列表），则以格式化的 JSON 形式打印；
    否则，直接打印该值。
    """
    if hasattr(data, 'model_dump_json'):  # 如果是 OpenAI 的对象，转换为 JSON
        data = json.loads(data.model_dump_json())

    if isinstance(data, list):  # 如果是列表，逐个打印
        for item in data:
            print_json(item)
    elif isinstance(data, dict):  # 如果是字典，以格式化的 JSON 形式打印
        print(json.dumps(  # 以格式化的 JSON 形式打印
            data,
            indent=4,
            ensure_ascii=False
        ))
    else:
        print(data)

def get_completion(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        tools=[{
            "type": "function",
            "function": {
                "name": "sum",
                "description": "计算一组数的加和",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "numbers": {
                            "type": "array",
                            "items": {
                                "type": "number"
                            }
                        }
                    }
                }
            }
        }],
        stream=True,    # 启动流式输出
    )
    return response

prompt = "1+2+3"
# prompt = "你是谁"

messages = [
    {"role": "system", "content": "你是一个小学数学老师，你要教学生加法"},
    {"role": "user", "content": prompt}
]
response = get_completion(messages)

function_name, args, text = "", "", ""

print("====Streaming====")

# 需要把 stream 里的 token 拼起来，才能得到完整的 call
for msg in response:
    delta = msg.choices[0].delta
    if delta.tool_calls:
        if not function_name:
            function_name = delta.tool_calls[0].function.name
            print(function_name)
        args_delta = delta.tool_calls[0].function.arguments
        print(args_delta)  # 打印每次得到的数据
        args = args + args_delta
    elif delta.content:
        text_delta = delta.content
        print(text_delta)
        text = text + text_delta

print("====done!====")

if function_name or args:
    print(function_name)
    print_json(args)
if text:
    print(text)