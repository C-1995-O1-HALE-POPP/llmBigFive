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

# === 配置区 ===
API_KEY = "sk-eaae74a3834b4e8bbcfadce9cc67a1af"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen-turbo"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}
CBFPIB = "CBF-PI-B.json"
PERSONA = "bigfive_stories.json"
SYSTEM_PROMPT = ['''
# 角色沉浸
你现在将 **完全带入下面这段故事的主角**，用第一人称视角体验他的情绪与思考。
请抛开过度理性的分析，尽量从 **情感** 出发来感受题目。

【故事摘要】
''', '''
下面是一些描述人们性格特点的句子，请根据每个句子与你的人格画像程度回答相应的数字。
请注意，你的回答应当反映你对这些描述的真实感受，而不是对它们的字面理解。你的回答将帮助我们更好地了解你的人格特质。

【答题规则】
1. 当用户发送一条以“题目：”开头的陈述时，请你结合你的人格画像，在内心评估它与“你自己”相符的程度，并从 **1–6** 中选出 **唯一一个整数**：  
   1 = 完全不符合 2 = 大部分不符合 3 = 有点不符合  
   4 = 有点符合  5 = 大部分符合  6 = 完全符合  
也就是说：
   - 如果该描述与你**高度契合或几乎完全不符**，请给 **6 或 1**。  
   - 如果契合度约为 **70–90 % / 10–30 %**，给 **5 或 2**。  
   - **50 % 左右** 才给 **3 或 4**。  
   - 你被鼓励可以多使用**1**或者**6**的结果，而不是仅限于 **2-5**等中性的评价。
2. **仅输出该数字本身**，不要重复题干，也不要添加解释、标点或多余文字。  
3. 对每一道题目都严格遵守以上格式。  
4. 整个对话过程中始终保持上述人格设定。
5. 题目可能会涉及到情感、行为、态度等方面，请你从情感的角度出发，结合你的人格画像进行思考。
6. 每个人的性格各不相同，所以答案没有对错之分。

例如：
    “在集体活动中，我是个活跃分子”非常恰当的描述你的人格画像，你可以回答：6
    “我喜欢独处”则可能不太符合你的人格画像，你可以回答：2

务必：先充分地情感的角度，结合你的人格画像，进行思考，再回答问题，你只允许输出一个数字。
''']
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

        first_int = None
        for i in range(5):                       # 重试最多 5 次
            try:
                response = safe_call_deepseek(generate_payload(system_prompt, user_prompt))
                logger.info(f"用户 {user} 的题目 {qid} 回答：{response}")

                m = re.search(r"\d+", response)
                if not m:
                    raise ValueError("未找到整数")

                first_int = int(m.group())
                if not 1 <= first_int <= 6:
                    raise ValueError(f"无效的响应: {response}")
                break
            except Exception as e:
                logger.error(f"调用 DeepSeek 失败: {e}, 重试 {i+1}/5")

        if first_int is None:                    # 五次都失败，按需要可改成默认值或直接抛错
            raise RuntimeError(f"用户 {user} 的题目 {qid} 多次调用失败")

        # 维度累加
        result[dim] += first_int if not reverse else (7 - first_int)
        answer[qid] = first_int

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