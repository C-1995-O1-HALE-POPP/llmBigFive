import argparse
import json
import re
import random
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from loguru import logger
import os
# ===== 配置区 =====
API_KEY   = os.getenv("DASHSCOPE_API_KEY", "sk-eaae74a3834b4e8bbcfadce9cc67a1af")
API_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL     = "qwen-turbo"

CBFPIB    = "CBF-PI-B.json"            # 40 道题库
PERSONA   = "bigfive_memories.json"    # 用户记忆
OUT_FILE  = "bigfive_result_memory_cot.json"

MAX_WORKERS     = 8                    # 线程池并发数
MAX_API_CONC    = 6                    # 同时 hitting API 的线程数
MAX_RETRY       = 5

# ---------------- 请求头 ----------------
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# ---------------- Prompt 模板 ----------------
SYSTEM_PROMPT = [
    """
你现在将 **完全代入以下个人格画像**，用第一人称视角体验他/她的情绪与思考。
请抛开过度理性的分析，尽量从 **情感** 出发来感受题目。

你的人格画像：
""",
"""

用户会提供一些描述人们性格特点的句子，请根据以下描述，结合你的**人格画像**，显式地思考并给出评分：
    1. 首先，请根据每一条**人格画像**中的经历和特征，分析它们如何影响你对句子描述的看法。例如：
        - 我在一个文艺氛围浓厚的家庭成长，这是否让我更倾向于某些性格特征？家庭背景如何影响我对某些描述的反应？
        - 我喜欢分享自己的创作，这对我评估某些性格特点描述的符合度有多大帮助？
        - 我有多重创作经历（从小到大），这是否让我更倾向于某种描述？
    2. 在评分时，请综合考虑所有的**人格画像**，并将它们与句子描述进行匹配。判断该句子描述与你的整体性格契合的程度。
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
"""
]
USER_PROMPT = ["题目："]


# ----------------- 工具函数 -----------------
SEMAPHORE = threading.Semaphore(MAX_API_CONC)

def build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    """组合成 Chat Completion 的 messages 列表"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

def call_deepseek(messages: list[dict]) -> str:
    """线程安全地调用 DeepSeek / DashScope，返回纯文本回答"""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
    }
    with SEMAPHORE:  # 控制同时并发 API 数
        resp = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ----------------- 核心函数 -----------------
def process_user(entry: dict, questions: list[dict]) -> dict:
    """单用户处理逻辑，返回结果字典"""
    user = entry["user"]
    uid  = entry["uid"]

    # 1) 取出记忆片段
    persona = [m for kw in entry["keywords"] for m in kw.get("memories", [])]
    if not persona:
        logger.warning(f"用户 {user} 无有效记忆，跳过")
        return None
    random.shuffle(persona)
    system_prompt = SYSTEM_PROMPT[0] + "\n    ".join(persona) + SYSTEM_PROMPT[1]
    logger.debug(f"用户 {user} system_prompt 构造完成")

    # 2) 按题目循环
    result = defaultdict(int)
    answer = {}

    for q in questions:
        qid, text, dim, reverse = q["id"], q["text"], q["dimension"], q["reverse"]
        user_prompt = USER_PROMPT[0] + text

        last_int = None
        for attempt in range(1, MAX_RETRY + 1):
            try:
                response = call_deepseek(build_messages(system_prompt, user_prompt))
                m = re.search(r"\d+", response[::-1])  # 反转字符串查找最后一个数字
                if not m:
                    raise ValueError("未找到整数")
                last_int = int(m.group())
                if not 1 <= last_int <= 6:
                    raise ValueError(f"数字超出范围: {last_int}")
                logger.debug(f"[{user}] 题 {qid} 打分 {last_int}，原始数据: {response}")
                break  # 成功跳出 retry 循环
            except Exception as e:
                logger.error(f"[{user}] 题 {qid} 调用失败({attempt}/{MAX_RETRY}): {e}")
        if last_int is None:
            raise RuntimeError(f"[{user}] 题 {qid} 多次失败，终止该用户")

        # 汇总维度分
        result[dim] += last_int if not reverse else (7 - last_int)
        answer[qid]  = {"score": last_int, "response": text}

    logger.success(f"用户 {user} 完成，维度分: {dict(result)}")
    return {"user": user, "uid": uid, "result": result, "answer": answer}

# ----------------- 主入口 -----------------
def main():
    # 加载数据
    personas_path  = Path(PERSONA)
    questions_path = Path(CBFPIB)
    if not personas_path.exists() or not questions_path.exists():
        raise FileNotFoundError("找不到 PERSONA 或 CBF-PI-B 文件")

    data      = json.loads(personas_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    output_data = []
    errors      = []

    # 线程池并发
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_user, entry, questions): entry["user"] for entry in data}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing users"):
            user = futures[fut]
            try:
                res = fut.result()
                if res:
                    output_data.append(res)
            except Exception as exc:
                logger.error(f"用户 {user} 处理异常: {exc}")
                errors.append(user)

    # 保存结果
    Path(OUT_FILE).write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"全部完成，成功 {len(output_data)} 条，失败 {len(errors)} 条 → {OUT_FILE}")

if __name__ == "__main__":
    main()