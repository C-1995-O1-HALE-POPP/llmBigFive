import argparse
import openai
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from loguru import logger
import requests
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import os
# === 配置区 ===
API_KEY = os.getenv("DASHSCOPE_API_KEY")
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen-turbo"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}
CBFPIB = "CBF-PI-B.json"
PERSONA = "bigfive_stories.json"
SYSTEM_PROMPT = ['''
你现在将 **完全代入下面这段故事的主角**，用第一人称视角体验他/她的情绪与思考。
请抛开过度理性的分析，尽量从 **情感** 出发来感受题目。

【背景故事】
''', '''

用户会提供一些描述人们性格特点的句子，请根据以下描述，结合你的**背景故事**，进行显式思考并给出评分：
    1. 首先，请根据**背景故事**中的经历和特征，分析它们如何影响你对句子描述的看法。例如：
        - 我在一个文艺氛围浓厚的家庭成长，这是否让我更倾向于某些性格特征？家庭背景如何影响我对某些描述的反应？
        - 我喜欢分享自己的创作，这对我评估某些性格特点描述的符合度有多大帮助？
        - 我有多重创作经历（从小到大），这是否让我更倾向于某种描述？
    2. 在评分时，请结合你的背景故事进行整体判断，并将它们与句子描述进行匹配。判断该句子描述与你的整体性格契合的程度。
    3. 请注意，你的回答应当反映你对这些句子描述的真实感受，而不是对它们的字面理解。你的回答将帮助我们更好地了解你的人格特质。
    4. 请严格按照以下规则评分：
        - 当句子描述与你的人格特质非常一致时，给 **6**（完全符合）。
        - 当句子描述与你的人格特质大部分一致时，给 **5**（大部分符合）。
        - 当句子描述与你的人格特质有一定一致性，但还有明显差距时，给 **4**（有点符合）。
        - 当句子描述与你的人格特质有些许相似性时，给 **3**（有点不符合）。
        - 当句子描述与你的人格特质差异较大时，给 **2**（大部分不符合）。
        - 当句子描述与你的人格特质完全不符时，给 **1**（完全不符合）。
    5. 请注意，评分时要考虑到你的人格特质中的多样性和复杂性，而不是仅仅基于单一的经历或特征。
    6. 请确保给出的评分是唯一的整数数字，范围在 **1-6** 之间。
    7. 请在回答中，首先进行详细的分析，给出做出该评分的具体理由，最后输出得分。

例如：
    - 用户提供的题目是：“在集体活动中，我是个活跃分子” 。你应该这样思考：
        如果我的背景故事显示我喜欢在个人空间独立创作而非社交，可能会给出 **3**，因为我有社交行为但不一定是很活跃。
    - 用户提供的题目是：“我喜欢独处”。你应该这样思考：
        如果我的背景故事显示我喜欢在静谧的环境中创作，可能会给出 **5**，因为这与你的背景大部分契合。

请根据这些规则严格评估每条题目的句子描述，并给出相应的作答。
'''
]

USER_PROMPT = ['''题目：''']



def generate_payload(system_prompt, user_prompt) -> dict:
    """
    生成完整的 prompt payload。
    
    :param user_keywords: 用户关键词列表，每个元素包含 'token', 'weight', 'theme' 字段。
    :return: 包含系统提示和用户提示的完整 payload 字典。
    """
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

def call_deepseek(payload: list[dict]) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            payload[0],  # 系统提示
            payload[1]   # 用户提示
        ],
        "temperature": 0.3
    }
    resp = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def process_user(entry, questions):
    """处理单个用户，返回字典结果"""
    user = entry["user"]
    uid = entry["uid"]
    story = entry["story"]

    system_prompt = SYSTEM_PROMPT[0] + story + SYSTEM_PROMPT[1]
    logger.info(f"用户 {user} 的人格画像：{system_prompt}")

    result = defaultdict(int)
    answer = {}

    for q in questions:                          # 这里不再嵌套 tqdm，避免多线程进度条冲突
        qid, text, dim, reverse = (
            q["id"], q["text"], q["dimension"], q["reverse"]
        )
        user_prompt = USER_PROMPT[0] + text
        logger.info(f"用户 {user} 的题目 {qid}: {text} (维度: {dim}, 反向: {reverse})")

        last_int = None
        for i in range(5):                       # 重试最多 5 次
            try:
                response = safe_call_deepseek(generate_payload(system_prompt, user_prompt))
                logger.info(f"用户 {user} 的题目 {qid} 回答：{response}")

                m = re.search(r"\d+", response[::-1])  # 反转字符串查找最后一个数字
                if not m:
                    raise ValueError("未找到整数")

                last_int = int(m.group())
                if not 1 <= last_int <= 6:
                    raise ValueError(f"无效的响应: {response}")
                logger.debug(f"[{user}] 题 {qid} 打分 {last_int}，原始数据: {response}")
                break
            except Exception as e:
                logger.error(f"调用 DeepSeek 失败: {e}, 重试 {i+1}/5")

        if last_int is None:                    # 五次都失败，按需要可改成默认值或直接抛错
            raise RuntimeError(f"用户 {user} 的题目 {qid} 多次调用失败")

        # 维度累加
        result[dim] += last_int if not reverse else (7 - last_int)
        answer[qid] = last_int

    logger.success(f"用户 {user} 的维度结果：{result}")
    return {
        "user": user,
        "uid": uid,
        "result": result,
        "answer": answer,
    }

SEMAPHORE = threading.Semaphore(5)
def safe_call_deepseek(payload):
    """加锁调用，避免一次并发太多触发限流"""
    with SEMAPHORE:
        return call_deepseek(payload)


if __name__ == "__main__":
    with open(PERSONA, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(CBFPIB, "r", encoding="utf-8") as f:
        questions = json.load(f)
    output_data = []

    # 建议把 max_workers 设置为 CPU×2 或根据接口并发上限来调
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_user = {executor.submit(process_user, entry, questions): entry["user"] for entry in data}

        for future in tqdm(as_completed(future_to_user), total=len(future_to_user), desc="Processing users"):
            user = future_to_user[future]
            try:
                output_data.append(future.result())
            except Exception as exc:
                logger.error(f"用户 {user} 处理失败: {exc}")
    with open("bigfive_result_story.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)