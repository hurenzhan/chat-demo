from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import requests
import json

amap_key = "6d672e6194caa3b639fccf2caf06c342"

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
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
        seed=1024,  # 随机种子保持不变，temperature 和 prompt 不变的情况下，输出就会不变
        tool_choice="auto",  # 默认值，由 GPT 自主决定返回 function call 还是返回文字回复。也可以强制要求必须调用指定的函数，详见官方文档
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_location_coordinate",
                    "description": "根据POI名称，获得POI的经纬度坐标",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "POI名称，必须是中文",
                            },
                            "city": {
                                "type": "string",
                                "description": "POI所在的城市名，必须是中文",
                            }
                        },
                        "required": ["location", "city"],
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_nearby_pois",
                    "description": "搜索给定坐标附近的poi",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "longitude": {
                                "type": "string",
                                "description": "中心点的经度",
                            },
                            "latitude": {
                                "type": "string",
                                "description": "中心点的纬度",
                            },
                            "keyword": {
                                "type": "string",
                                "description": "目标poi的关键字",
                            }
                        },
                        "required": ["longitude", "latitude", "keyword"],
                    }
                }
            }
        ],
    )
    return response.choices[0].message


def get_location_coordinate(location, city):
    url = f"https://restapi.amap.com/v5/place/text?key={amap_key}&keywords={location}&region={city}"
    print(url)

    r = requests.get(url)
    result = r.json()

    if "pois" in result and result["pois"]:
        return result["pois"][0]
    return None


def search_nearby_pois(longitude, latitude, keyword):
    url = f"https://restapi.amap.com/v5/place/around?key={amap_key}&keywords={keyword}&location={longitude},{latitude}"
    print(url)
    r = requests.get(url)
    result = r.json()
    ans = ""
    if "pois" in result and result["pois"]:
        for i in range(min(3, len(result["pois"]))):
            name = result["pois"][i]["name"]
            address = result["pois"][i]["address"]
            distance = result["pois"][i]["distance"]
            ans += f"{name}\n{address}\n距离：{distance}米\n\n"
    return ans


prompt = "我想在南京九龙湖附近找vivo专卖店"

messages = [
    {"role": "system", "content": "你是一个地图通，你可以找到任何地址。"},
    {"role": "user", "content": prompt}
]

response = get_completion(messages)
messages.append(response)  # 把大模型的回复加入到对话中
# print("=====GPT回复=====")
# print_json(response)

while (response.tool_calls is not None):
    # 1106 版新模型支持一次返回多个函数调用请求，所以要考虑到这种情况
    for tool_call in response.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = None
        # print("函数参数展开：")
        # print_json(args)

        if (tool_call.function.name == "get_location_coordinate"):
            print("Call: get_location_coordinate")
            result = get_location_coordinate(**args)
        elif tool_call.function.name == "search_nearby_pois":
            print("Call: search_nearby_pois")
            result = search_nearby_pois(**args)

        # print("=====函数返回=====")
        # print_json(result)

        messages.append({
            "tool_call_id": tool_call.id,  # 用于标识函数调用的 ID
            "role": "tool",
            "name": tool_call.function.name,
            "content": str(result)  # 数值result 必须转成字符串
        })

    response = get_completion(messages)
    messages.append(response)  # 把大模型的回复加入到对话中

print("=====对话记录=====")
print_json(messages)
