from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

client = OpenAI()


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

# 案例：客服质检
#
# 任务本质是检查客服与用户的对话是否有不合规的地方
#
#     质检是电信运营商和金融券商大规模使用的一项技术
#     每个涉及到服务合规的检查点称为一个质检项
#
# 我们选一个质检项，产品信息准确性，来演示思维链的作用：
#
#     当向用户介绍流量套餐产品时，客服人员必须准确提及产品名称、月费价格、月流量总量、适用条件（如有）
#     上述信息缺失一项或多项，或信息与事实不符，都算信息不准确

# 规则
instruction = """
给定一段用户与手机流量套餐客服的对话，。
你的任务是判断客服介绍产品信息的准确性：

当向用户介绍流量套餐产品时，客服人员必须准确提及产品名称、月费价格和月流量总量 上述信息缺失一项或多项，或信息与实时不符，都算信息不准确

已知产品包括：

经济套餐：月费50元，月流量10G
畅游套餐：月费180元，月流量100G
无限套餐：月费300元，月流量1000G
校园套餐：月费150元，月流量200G，限在校学生办理
"""

# 输出描述
output_format = """
如果信息准确，输出：Y
如果信息不准确，输出：N
"""

context = """
用户：你们有什么流量大的套餐
客服：您好，我们现在正在推广无限套餐，每月300元就可以享受1000G流量，您感兴趣吗
"""

cot = ""

# 思维链 (Chain of Thoughts, CoT)
# 有人在提问时以「Let’s think step by step」开头，结果发现 AI 会把问题分解成多个步骤，然后逐步解决，使得输出的结果更加准确。
# cot = "请一步一步分析以下对话"

# -------------------------------------------
# prompt = f"""
# {instruction}
#
# {output_format}
#
# {cot}
#
# 对话记录：
# {context}
# """
#
# response = get_completion(prompt)
# print(response)
# -------------------------------------------

# 自洽性 (Self-Consistency)
#  同样 prompt 跑多次，通过投票选出最终结果
for _ in range(5):
    prompt = f"{instruction}\n\n{output_format}\n\n请一步一步分析:\n{context}"
    print(f"------第{_+1}次------")
    response = get_completion(prompt)
    print(response)
