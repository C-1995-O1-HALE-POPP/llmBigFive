import argparse
import json
import re
import random
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt_generator import generate_system_prompt

import requests
from tqdm import tqdm
from loguru import logger
import os
# ===== 配置区 =====
API_KEY   = os.getenv("DASHSCOPE_API_KEY")
API_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

CBFPIB    = "CBF-PI-B.json"            # 40 道题库

MAX_WORKERS     = 8                    # 线程池并发数
MAX_API_CONC    = 6                    # 同时 hitting API 的线程数
MAX_RETRY       = 5
SEMAPHORE = threading.Semaphore(MAX_API_CONC)

model, repeat, system_prompt_raw = "qwen_turbo", 1, ""

# ---------------- 请求头 ----------------
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# ---------------- Prompt 模板 ----------------
USER_PROMPT = ["题目："]

# ----------------- 工具函数 -----------------


def build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    """组合成 Chat Completion 的 messages 列表"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

def call_deepseek(messages: list[dict]) -> str:
    """线程安全地调用 DeepSeek / DashScope，返回纯文本回答"""
    global model
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
    }
    with SEMAPHORE:  # 控制同时并发 API 数
        resp = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ----------------- 核心函数 -----------------
def process_user(entry: dict, questions: list[dict], seed: int = 42) -> dict:
    """单用户处理逻辑，返回结果字典"""
    user = entry["user"]
    uid  = entry["uid"]

    global system_prompt_raw, repeat
    # 1) 取出记忆片段
    persona = [m for kw in entry.get("keywords", []) for m in kw.get("memories", [])]
    answer, result = {}, {}
    for q in questions:
        answer[q["id"]], result[q["dimension"]] = {}, defaultdict(int)

    for i in range(repeat):
        if not persona:
            system_prompt = system_prompt_raw[1]
        else:
            random.seed(seed)
            random.shuffle(persona)
            system_prompt = system_prompt_raw[0] + "\n    ".join(persona) + system_prompt_raw[1]
        # logger.debug(f"用户 {user} system_prompt 构造完成：{system_prompt}")

        # 2) 按题目循环
        for q in questions:
            qid, text, dim, reverse = q["id"], q["text"], q["dimension"], q["reverse"]
            user_prompt = USER_PROMPT[0] + text

            last_int = None
            for attempt in range(1, MAX_RETRY + 1):
                try:
                    response = call_deepseek(build_messages(system_prompt, user_prompt))
                    m = re.search(r"\d+", response[::-1])
                    if not m:
                        raise ValueError("未找到整数")
                    last_int = int(m.group())
                    logger.debug(f"[{user}] 题 {qid} 第 {i} 次回答：{response} → {last_int}")
                    if not 1 <= last_int <= 6:
                        raise ValueError(f"数字超出范围: {last_int}")
                    break  # 成功跳出 retry 循环
                except Exception as e:
                    logger.error(f"[{user}] 题 {qid} 第 {i} 次回答调用失败({attempt}/{MAX_RETRY}): {e}。原始回答: {response if 'response' in locals() else '无'}")
            if last_int is None:
                raise RuntimeError(f"[{user}] 题 {qid} 第 {i} 次回答多次失败，终止该用户")

            # 汇总维度分
            result[dim][i] += last_int if not reverse else (7 - last_int)
            answer[qid][i]  = {"answer": last_int, "text": response}

    logger.success(f"用户 {user} 完成，维度分: {dict(result)}")
    return {"user": user, "uid": uid, "result": result, "answer": answer}

# ----------------- 主入口 -----------------
def main():
    argparse.ArgumentParser(description="CBF-PI-B Memory Zero-shot Processing")
    parser = argparse.ArgumentParser(description="CBF-PI-B Memory Zero-shot Processing")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认为 42")
    parser.add_argument("--model", type=str, required=True ,help="模型名称，例如 qwen-turbo")
    parser.add_argument("--data-type", type=str, default="memory", choices=["memory", "story"], help="处理类型，默认为 memory")
    parser.add_argument("--cot", action="store_true", help="是否启用 Chain of Thought (CoT) 模式")
    parser.add_argument("--zeroshot", action="store_true", help="是否启用zeroshot")
    parser.add_argument("--repeat", type=int, default=1, help="重复次数，默认为 1")
    args = parser.parse_args()

    # 设置全局变量
    global model, repeat, system_prompt_raw
    model, data_type, cot, zeroshot, repeat = args.model, args.data_type, args.cot, args.zeroshot, args.repeat
    seed = args.seed
    system_prompt_raw = generate_system_prompt(data_type, cot, zeroshot)
    logger.info(f"使用模型: {model}, 数据类型: {data_type}, CoT: {cot}, Zero-shot: {zeroshot}, 重复次数: {repeat}")

    # 文件路径
    if not Path("results").exists or not Path("results").is_dir():
        os.mkdir("results")
    outfile_path = f'''bigfive_result_{data_type}{"_cot" if cot else ""}{"_zeroshot" if zeroshot else ""}_repeat{repeat}.json'''
    outfile_path = Path.joinpath(Path("results"), Path(outfile_path))
    personas_path  = Path("bigfive_memories.json" if data_type == "memory" else "bigfive_stories.json") 
    questions_path = Path(CBFPIB)


    data      = json.loads(personas_path.read_text(encoding="utf-8"))
    data.append({"user": "baseline", "uid": -1, "story": ""})

    outfile = json.loads(outfile_path.read_text(encoding="utf-8")) if outfile_path.exists() else []
    data_filtered = [i for i in data if i["user"] not in [j["user"] for j in outfile]]  # 去重
    data = data_filtered if data_filtered else data
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    output_data = outfile
    errors      = []

    # 线程池并发
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_user, entry, questions, seed): entry["user"] for entry in data}

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
    outfile_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"全部完成，成功 {len(output_data)} 条，失败 {len(errors)} 条 → {outfile_path}")

if __name__ == "__main__":
    main()