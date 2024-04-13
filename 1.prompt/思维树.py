import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

client = OpenAI()


# 只有 gpt-4 能跑动思维树。实验室不支持 gpt-4，自行实验请在本地运行


def get_completion(prompt, model="gpt-4-turbo", temperature=0, response_format="text"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # 模型输出的随机性，0 表示随机性最小
        response_format={"type": response_format},
    )
    return response.choices[0].message.content


# 思维树（Tree-of-thought, ToT）
#
# 在思维链的每一步，采样多个分支
# 拓扑展开成一棵思维树
# 判断每个分支的任务完成度，以便进行启发式搜索
# 设计搜索算法
# 判断叶子节点的任务完成的正确性
#
# 案例：指标解读，项目推荐并说明依据
# 小明 100 米跑成绩：10.5 秒，1500 米跑成绩：3 分 20 秒，铅球成绩：12 米。他适合参加哪些搏击运动训练。

# 能力分析
def performance_analyser(text):
    prompt = f"{text}\n请根据以上成绩，分析候选人在速度、耐力、力量三方面素质的分档。分档包括：强（3），中（2），弱（1）三档。\
                \n以JSON格式输出，其中key为素质名，value为以数值表示的分档。"

    response = get_completion(prompt, response_format="json_object")

    print(response)  # {'速度': 3, '耐力': 2, '力量': 2}

    return json.loads(response)


# 运动推荐
def possible_sports(talent, category):
    prompt = f"""
        需要{talent}强的{category}运动有哪些。给出10个例子，以array形式输出。确保输出能由json.loads解析。"""

    response = get_completion(prompt, temperature=0.8,
                              response_format="json_object")

    print(response)  # {'speed_intense_martial_arts': ['拳击', '跆拳道', '空手道', '泰拳', '快速摔跤', '散打', '剑道', '击剑', '巴西柔术', 'MMA（综合格斗）']}

    # return json.loads(response)
    return next(iter(json.loads(response).values()))


# 评估运动对某项素质的要求
def evaluate(sports, talent, value):
    prompt = f"分析{sports}运动对{talent}方面素质的要求: 强（3），中（2），弱（1）。\
                \n直接输出挡位数字。输出只包含数字。"

    response = get_completion(prompt)

    val = int(response)

    print(f"{sports}: {talent} {val} {value >= val}")

    return value >= val

# 生成报告
def report_generator(name, performance, talents, sports):
    level = ['弱', '中', '强']

    _talents = {k: level[v - 1] for k, v in talents.items()}

    prompt = f"已知{name}{performance}\n身体素质：\
        {_talents}。\n生成一篇{name}适合{sports}训练的分析报告。"

    response = get_completion(prompt, model="gpt-3.5-turbo")

    return response

name = "小明"
performance = "100米跑成绩：10.5秒，1500米跑成绩：3分20秒，铅球成绩：12米。"
category = "搏击"

talents = performance_analyser(name+performance)
print("===talents===")
print(talents)

cache = set()
# 深度优先

# 第一层节点
for k, v in talents.items():
    if v < 3:  # 剪枝
        continue
    leafs = possible_sports(k, category)
    print(f"==={k} leafs===")
    print(leafs)
    # 第二层节点
    for sports in leafs:
        if sports in cache:
            continue
        cache.add(sports)
        suitable = True
        for t, p in talents.items():
            if t == k:
                continue
            # 第三层节点
            if not evaluate(sports, t, p):  # 剪枝
                suitable = False
                break
        if suitable:
            report = report_generator(name, performance, talents, sports)
            print("****")
            print(report)
            print("****")
