from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import json

_ = load_dotenv(find_dotenv())

client = OpenAI()


def get_completion(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,  # 模型输出的随机性，0 表示随机性最小
        tools=[{  # 用 JSON 描述函数。可以定义多个。由大模型决定调用谁。也可能都不调用
            "type": "function",
            "function": {
                "name": "sum",  # 函数名
                "description": "加法器，计算一组数的和",  # 函数的描述
                "parameters": {  # 函数的参数
                    "type": "object",  # 参数的类型
                    "properties": {  # 参数的属性
                        "numbers": {  # 参数名
                            "type": "array",  # 参数的类型
                            "items": {  # 数组的元素
                                "type": "number"  # 元素的类型
                            }
                        }
                    }
                }
            }
        }],
    )
    return response.choices[0].message


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


prompt = "Tell me the sum of 1, 2, 3, 4, 5, 6, 7, 8, 9, 10."
# prompt = "桌上有 2 个苹果，四个桃子和 3 本书，一共有几个水果？"
# prompt = "1+2+3...+99+100"
# prompt = "1024 乘以 1024 是多少？"   # Tools 里没有定义乘法，会怎样？
# prompt = "太阳从哪边升起？"           # 不需要算加法，会怎样？

messages = [
    {"role": "system", "content": "你是一个数学家"},
    {"role": "user", "content": prompt}
]
response = get_completion(messages)

# 把大模型的回复加入到对话历史中
messages.append(response)

print("=====GPT回复=====")
print_json(response)
