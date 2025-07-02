import json, itertools, requests
from collections import defaultdict
from tqdm import tqdm
from loguru import logger

SYSTEM_PROMPT = """你是一个心理学家，擅长分析用户的个性特征。你善于根据有限线索提炼人格特质，并用流畅、自然的中文第一人称完成画像。"""
USER_PROMPT = ["""请根据下面按主题关键词组合，为「我」生成一系列的简短的中文人格自我画像描述。
要求：
1. 只输出一句话，内容完整，不要输出多余的内容；
2. 每句话尽量融入关键词中的概念（可同义替换）；
3. 语气负面、积极、中性皆可，但要真实连贯；
4. 不要写列表、不要出现“关键词”“主题”字样。
4. **输出格式必须是纯 JSON 字符串**，内容结构如下  
   ```json
   {
     "description": [
       "第一句……",
       "第二句……"
     ]
   }
   ```
仅包含 description 这一键，且务必保证合法可解析；不得输出任何额外文本（包括注释、前后缀、代码块标记等）。
例如，我提供的{主题, 关键词}是{爱好和娱乐, 创意}，指导你生成2句描述性语句。你只需要输出以下内容：
    ```json
    {
      "description": [
        "我喜欢通过创意活动来放松自己，享受生活中的乐趣和新鲜感。",
        "在我的日常生活中，创意是我表达自我的重要方式，它让我感到充实和快乐。"
      ]
    }
    ```

现在，我提供的{主题, 关键词}是{""", """}，你应该输出**""""""**句描述性语句。你的输出是："""]

import os
API_KEY = os.getenv("DASHSCOPE_API_KEY")
K = 10  # 保留关键词数
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen-turbo"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}
def generate_payload(theme, keyword):
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"{USER_PROMPT[0]}{theme}, {keyword}{USER_PROMPT[1]}"
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
        except requests.exceptions.HTTPError as e:
            logger.error(f"请求失败: {e}, 重试 {i+1}/5")

    return resp["choices"][0]["message"]["content"]

if __name__ == "__main__":
    with open("bigfive_prompt_payload.json") as f:   # 你的那份原始列表
        data = json.load(f)

    valid_kw = []
    for entry in data:
        user = entry["user"]
        keywords = entry["keywords"]
        top_keywords = sorted(keywords, key=lambda x: x["weight"], reverse=True)[:K]
        valid_kw.append({
            "user": user,
            "keywords": top_keywords
        })
    logger.success(f"提取到 {len(valid_kw)} 个用户的关键词")
    for entry in valid_kw:
        logger.info(f"用户 {entry['user']} 的关键词：{entry['keywords']}") 
        for kw in tqdm(entry["keywords"]):
            theme = kw["context"]
            token = kw["token"]
            payload = generate_payload(theme, token)
            try:
                response = call_deepseek(payload)
            except Exception as e:
                logger.error(f"调用 DeepSeek 失败: {e}")
            logger.info(f"用户 {entry['user']} 的 {theme}, {token} 生成的描述：{response}")
            kw["description"] = response
    with open("bigfive_persona.json", "w", encoding="utf-8") as f:
        json.dump(valid_kw, f, ensure_ascii=False, indent=2)
    logger.success("生成的人格画像已保存到 bigfive_persona.json")
    
