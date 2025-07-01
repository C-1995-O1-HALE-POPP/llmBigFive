import json
from pathlib import Path
from typing import List, Dict
import requests
from loguru import logger
from tqdm import tqdm
import random
import os

# ============ 配置区 ============
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen-turbo"
API_KEY = os.getenv("DASHSCOPE_API_KEY")
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

SOURCE_FILE = "bigfive_story.json"    # 上一步增量生成的文件
OUTPUT_JSON = Path("stories.json")  # 故事输出文件

# =================================

def sort_memories(user_kw: List[Dict]) -> List[Dict]:
    flat = []
    for kw in user_kw:
        for sent in kw.get("memories", []):
            flat.append(sent)
    random.shuffle(flat)  # 打乱顺序，增加多样性
    return flat


def build_messages(memories_sorted: List[str]) -> List[Dict]:
    """生成 Chat Completions 所需的 messages 列表。"""
    # System role
    system_msg = {
        "role": "system",
        "content": (
            "你是一位具有文学创作力的作家兼心理叙事者。你擅长将零散的个人记忆编织成情感丰沛、结构完整的第一人称故事，"
            "同时准确保留每条记忆的原句细节。"
        )
    }
    # 把全部句子 + 权重做成 bullets 传给模型
    bullets = "\n".join([
        f"{m}. {memories_sorted[m]}" for m in range(len(memories_sorted))
    ])

    user_prompt = f"""
以下是我的若干记忆片段（共 {len(memories_sorted)} 条）：
{bullets}

# 任务
1. 请将这些记忆**全部**编写成一篇情感鲜明、正式书面语的第一人称故事。
2. **保留每条记忆原句**，并且仅出现一次；你可以在它们前后自由添加描写和过渡，但是请注意情节的连贯。
3. 故事要有开头、发展、高潮与收束，展现我在不同场景中的人格层次与情感转折。
4. 不要列出编号，也不要出现“权重”“编号”等技术性词汇。
"""
    user_msg = {"role": "user", "content": user_prompt}
    return [system_msg, user_msg]


def call_llm(messages: List[Dict]) -> str:
    """与 Qwen/LLM 对话，返回故事文本。"""
    logger.info(messages)
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "enable_thinking": False
    }
    resp = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload), timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def load_stories() -> List[Dict]:
    if OUTPUT_JSON.exists():
        return json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    return []


def save_stories(stories: List[Dict]):
    OUTPUT_JSON.write_text(json.dumps(stories, ensure_ascii=False, indent=2), encoding="utf-8")


def story_already_done(stories: List[Dict], uid: str | int) -> bool:
    return any(s["uid"] == str(uid) or s["uid"] == uid for s in stories)


# ---------- 主流程 ----------

def main():
    raw_data = json.loads(Path(SOURCE_FILE).read_text(encoding="utf-8"))
    stories = load_stories()

    for user in tqdm(raw_data, desc="Writing stories"):
        if story_already_done(stories, user["uid"]):
            logger.info(f"用户 {user['uid']} 已存在，跳过 …")
            continue

        try:
            memories_sorted = sort_memories(user["keywords"])
            messages = build_messages(memories_sorted)
            story_text = call_llm(messages)

            record = {
                "uid": user["uid"],
                "user": user.get("user", ""),
                "story": story_text,
            }
            stories.append(record)
            save_stories(stories)  # 立刻写入磁盘
            logger.success(f"用户 {user['uid']} 故事已保存")
        except Exception as e:
            logger.error(f"用户 {user['uid']} 生成失败: {e}")

    logger.info("全部处理完毕！")


if __name__ == "__main__":
    main()
