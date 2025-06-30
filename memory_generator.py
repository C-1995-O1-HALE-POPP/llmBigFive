import json, itertools, requests
from collections import defaultdict
from tqdm import tqdm
from loguru import logger
import re

OUTPUT_PATH = "bigfive_story.json"  # 输出文件路径
SYSTEM_PROMPT = """
你是一位擅长提炼人物记忆片段的心理学家。
根据给定的「主题」与「关键词」，用自然流畅的中文第一人称，
为“我”写出 1-10 句个性化记忆。要求：
1. 总共输出 1-10 句完整的话（由调用端决定具体句数）；每句各自独立， 不出现内容上的重合。
2. 每句都要融入关键词（可同义替换）并包含能体现个人经历的**具体细节**；
3. 各句内容不得相互重复或在细节上雷同；
4. 语气可正面、负面或中性，但要真实连贯；
5. 不得出现“关键词”“主题”等字样；
4. **输出格式必须是纯 JSON 字符串**，内容结构如下  
   ```json
   {
     "memories": [
       "第一句……",
       "第二句……"
     ]
   }
   ```
仅包含 memories 这一键，且务必保证合法可解析；不得输出任何额外文本（包括注释、前后缀、代码块标记等）。
"""

# USER_PROMPT 分成两段，方便动态插入主题、关键词和目标句数 {n}
USER_PROMPT = [
    """请根据下面的「主题, 关键词」为“我”生成一系列短记忆，每句话的内容应该互相独立。
示例：生成3句有关{学业, 物理竞赛}的记忆。你应该回答：
   ```json
   {
     "memories": [
       "我擅长物理，在高中时期曾夺得省级物理竞赛一等奖。",
       "我在大学时选择了物理专业，深入研究了量子力学。",
       "我攻读了物理学硕士，之后在一家科研机构从事量子计算研究。"
     ]
   }
   ```

现在请你输出""", """句记忆，对应的的「主题, 关键词」是{""",
    """}。请输出："""
]

API_KEY = "sk-eaae74a3834b4e8bbcfadce9cc67a1af"
K = 10  # 保留关键词数
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen-turbo"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def pharse_json(json_str: str) -> dict:
    """
    尝试解析 JSON 字符串，若失败则返回空字典。
    """
    clean_json = re.sub(r'^```json\n|\n```$', '', json_str, flags=re.S)

    data = json.loads(clean_json)   # 解析
    return data["memories"]

    
def generate_payload(theme, keyword, n):
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"{USER_PROMPT[0]}{n}{USER_PROMPT[1]}{theme}, {keyword}{USER_PROMPT[2]}"
        }
    ]

def call_deepseek(payload: list[dict]) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            payload[0],  # 系统提示
            payload[1]   # 用户提示
        ],
        "enable_thinking": False
       #  "temperature": 0.3
    }
    for i in range(5):
        try:
            resp = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
            resp.raise_for_status()
            resp = resp.json()
            result = resp["choices"][0]["message"]["content"]
            result_list = pharse_json(result)
            if not result_list:
                logger.error(f"解析结果失败: {result}")
                continue
        except requests.exceptions.HTTPError as e:
            logger.error(f"请求失败: {e}, 重试 {i+1}/5")
    return result_list


def save_incremental(data_so_far: list[dict]):
    """把当前进度写入 OUTPUT_PATH。每次覆盖写，防止进程崩溃时丢失全部成果。"""
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(data_so_far, f, ensure_ascii=False, indent=2)
        logger.debug("已增量写入当前进度 …")
    except OSError as e:
        logger.error(f"写文件失败: {e}")


if __name__ == "__main__":
    with open("bigfive_prompt_payload.json") as f:   # 你的那份原始列表
        data = json.load(f)

    valid_kw = []
    for entry in data:
        user = entry["user"]
        uid = entry["uid"]
        keywords = entry["keywords"]
        top_keywords = sorted(keywords, key=lambda x: x["weight"], reverse=True)[:K]
        valid_kw.append({
            "user": user,
            "uid": uid,
            "keywords": top_keywords
        })
    logger.success(f"提取到 {len(valid_kw)} 个用户的关键词")
    for entry in valid_kw:
        logger.info(f"用户 {entry['user']} 的关键词：{entry['keywords']}") 
        for i in tqdm(range(len(entry["keywords"]))):
            n = len(entry["keywords"]) - i  # 剩余关键词数
            kw = entry["keywords"][i]
            theme = kw["context"]
            token = kw["token"]

            payload = generate_payload(theme, token, n)
            try:
                response = call_deepseek(payload)
            except Exception as e:
                logger.error(f"调用 DeepSeek 失败: {e}")
            logger.info(f"用户 {entry['user']} 的 {theme}, {token} 生成的描述：{response}")
            kw["memories"] = response
            save_incremental(valid_kw)
    logger.success("生成的人格画像已保存到 bigfive_story.json")
    
