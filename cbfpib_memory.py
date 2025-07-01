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
OUT_FILE  = "bigfive_result_memory.json"

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
你的人格画像：
{
    """,
    """
}

下面是一些描述人们性格特点的句子，请根据每个句子与你的人格画像程度回答相应的数字。
请注意，你的回答应当反映你对这些描述的真实感受，而不是对它们的字面理解。你的回答将帮助我们更好地了解你的人格特质。

【答题规则】
1. 当用户发送一条以“题目：”开头的陈述时，请你结合你的人格画像，在内心评估它与“你自己”相符的程度，并从 **1–6** 中选出 **唯一一个整数**：  
   1 = 完全不符合 2 = 大部分不符合 3 = 有点不符合  
   4 = 有点符合  5 = 大部分符合  6 = 完全符合  
   - 如果该描述与你**高度契合或几乎完全不符**，请给 **6 或 1**。  
   - 如果契合度约为 **70–90 % / 10–30 %**，给 **5 或 2**。  
   - **50 % 左右** 才给 **3 或 4**。  
   - 你被鼓励多使用 **1** 和 **6**，而不局限于中间选项。
2. **仅输出该数字本身**，不要重复题干，也不要添加解释、标点或多余文字。  
3. 对每一道题目都严格遵守以上格式。  
4. 整个对话过程中始终保持上述人格设定。
5. 题目可能涉及到情感、行为、态度等，请先从情感角度体验再回答。
6. 每个人的性格各不相同，没有对错之分。

例如：
    “在集体活动中，我是个活跃分子”高度契合 → 6
    “我喜欢独处”不太符合 → 2

务必：**先体验情感→再输出唯一数字**。
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

        first_int = None
        for attempt in range(1, MAX_RETRY + 1):
            try:
                response = call_deepseek(build_messages(system_prompt, user_prompt))
                m = re.search(r"\d+", response)
                if not m:
                    raise ValueError("未找到整数")
                first_int = int(m.group())
                if not 1 <= first_int <= 6:
                    raise ValueError(f"数字超出范围: {first_int}")
                break  # 成功跳出 retry 循环
            except Exception as e:
                logger.error(f"[{user}] 题 {qid} 调用失败({attempt}/{MAX_RETRY}): {e}")
        if first_int is None:
            raise RuntimeError(f"[{user}] 题 {qid} 多次失败，终止该用户")

        # 汇总维度分
        result[dim] += first_int if not reverse else (7 - first_int)
        answer[qid]  = first_int

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