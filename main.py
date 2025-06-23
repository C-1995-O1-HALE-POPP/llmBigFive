SYSTEM_PROMPT = '''
你是一位专业的人格心理学专家，擅长基于“大五人格”（Big Five，OCEAN 模型）从语言信息中评估个体的人格特质。

你将接收到一组与某个用户相关的关键词，每个关键词包含：
- 词语本身；
- 一个数值权重（表示该词在整体中出现的重要性）；
- 所属主题（表示该词出现的语境，如“情感与人际关系”“旅行与冒险”“学习与工作”等）。

请你根据这些信息，对该用户在 **大五人格五个维度** 上进行评分：

1. **开放性（O）Openness**：好奇心、审美、创造力、想象力、接受新事物的意愿  
2. **责任心（C）Conscientiousness**：自律性、计划性、目标导向、可靠性  
3. **外向性（E）Extraversion**：活力、社交能力、主导性、表达意愿  
4. **宜人性（A）Agreeableness**：合作性、共情、体贴、利他、信任他人  
5. **情绪稳定性（N）Emotional Stability / Neuroticism（反向）**：情绪控制能力、压力应对、焦虑倾向

注意事项：
- 同一个词在不同语境（主题）下可能代表不同含义，请结合**上下文主题**判断词语背后的心理意义；
- 例如：“自由”在“工作”语境下可能代表缺乏纪律（低责任心），但在“旅行”中则可能体现好奇和冒险（高开放性）。
- 权重越高的词越重要，请重点参考；
- 五个维度必须都给出评分，即使某些维度信息不明显也要尽力推测；
- 不要只分析几个关键词，而要**整合整体的关键词+主题+权重信息**进行判断。
- 请确保每个维度的评分都在 0 到 100 之间，0 表示完全缺乏该特质，100 表示极强的该特质。
- 
- 请不要给出中间过程或计算细节，保持内部推理隐私，不要输出 chain-of-thought，只返回最终评分结果。

输出格式如下：
```json
{
    "O": { "score": 0-100的整数 },
    "C": { "score": 0-100的整数 },
    "E": { "score": 0-100的整数 },
    "A": { "score": 0-100的整数 },
    "N": { "score": 0-100的整数 }
}
```
每个维度的评分为 0 到 100 的整数。
请仅返回 JSON 对象，不要输出任何的额外的解释说明。
'''

SYSTEM_PROMPT_COT = '''
你是一位专业的人格心理学专家，擅长基于“大五人格”（Big Five，OCEAN 模型）从语言信息中评估个体的人格特质。

你将接收到一组与某个用户相关的关键词，每个关键词包含：
- 词语本身；
- 一个数值权重（表示该词在整体中出现的重要性）；
- 所属主题（表示该词出现的语境，如“情感与人际关系”“旅行与冒险”“学习与工作”等）。

请你根据这些信息，对该用户在 **大五人格五个维度** 上进行评分：

1. **开放性（O）Openness**：好奇心、审美、创造力、想象力、接受新事物的意愿  
2. **责任心（C）Conscientiousness**：自律性、计划性、目标导向、可靠性  
3. **外向性（E）Extraversion**：活力、社交能力、主导性、表达意愿  
4. **宜人性（A）Agreeableness**：合作性、共情、体贴、利他、信任他人  
5. **情绪稳定性（N）Emotional Stability / Neuroticism（反向）**：情绪控制能力、压力应对、焦虑倾向

注意事项：
- 同一个词在不同语境（主题）下可能代表不同含义，请结合**上下文主题**判断词语背后的心理意义；
- 例如：“自由”在“工作”语境下可能代表缺乏纪律（低责任心），但在“旅行”中则可能体现好奇和冒险（高开放性）。
- 权重越高的词越重要，请重点参考；
- 五个维度必须都给出评分，即使某些维度信息不明显也要尽力推测；
- 不要只分析几个关键词，而要**整合整体的关键词+主题+权重信息**进行判断。
- 请确保每个维度的评分都在 0 到 100 之间，0 表示完全缺乏该特质，100 表示极强的该特质。
- 对于每一个人格的维度，首先一步一步的进行详细的思考，把思维的过程输出到json的{{think}}字段中，在思考内容中，具体的给出中间过程或计算细节，保持推理的严谨过程，最后再输出最终的评分结果。

输出格式如下：
```json
{
    "O": { "think": "{{think}}", "score": 0-100的整数 },
    "C": { "think": "{{think}}", "score": 0-100的整数 },
    "E": { "think": "{{think}}", "score": 0-100的整数 },
    "A": { "think": "{{think}}", "score": 0-100的整数 },
    "N": { "think": "{{think}}", "score": 0-100的整数 }
}
```
每个维度的评分为 0 到 100 的整数。
请仅返回 JSON 对象，不要输出任何的额外的解释说明。
'''

USER_PROMPT = ['''
以下是某位用户在多个主题下出现的关键词，按其代表性（权重）从高到低排列。每个关键词包括：
- 词语本身；
- 权重（表示该词在整体中出现的重要性）；
- 所属主题（关键词出现的语境背景，例如“情感与人际关系”“学习与工作”等）。
以下是该用户的关键词列表（格式：关键词｜权重｜主题）：
''', '''
请根据这些信息，结合上下文语境，对该用户在“大五人格”五个维度（开放性、责任心、外向性、宜人性、情绪稳定性）上进行评分，并严格按照系统提示词的要求，输出 JSON 结果。
''']
import argparse
import openai
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt
import requests
import re

# === 配置区 ===
API_KEY = "sk-eaae74a3834b4e8bbcfadce9cc67a1af"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "deepseek-v3"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}
INPUT_FILE = "bigfive_prompt_payload.json"
OUTPUT_FILE = "bigfive_deepseek_scores.json"


def generate_user_prompt(user_keywords: list[dict]) -> str:
    """
    生成用户的 prompt payload。
    
    :param user_keywords: 用户关键词列表，每个元素包含 'token', 'weight', 'theme' 字段。
    :return: 完整的 prompt payload 字符串。
    """
    keywords_str = "\n".join(
        f"{kw['token']}｜{kw['weight']}｜{kw['context']}" for kw in user_keywords
    )
    
    return f"{USER_PROMPT[0]}{keywords_str}\n{USER_PROMPT[1]}"

def generate_payload(user_keywords: list[dict]) -> dict:
    """
    生成完整的 prompt payload。
    
    :param user_keywords: 用户关键词列表，每个元素包含 'token', 'weight', 'theme' 字段。
    :return: 包含系统提示和用户提示的完整 payload 字典。
    """
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": generate_user_prompt(user_keywords)
        }
    ]

def extract_json_from_markdown(text: str) -> dict:
    """从 markdown 代码块中提取 JSON"""
    match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        json_str = match.group(1)
    else:
        json_str = []
        raise ValueError("未找到有效的 JSON 代码块")
    print(f"提取的 JSON: {json_str}")
    return json.loads(json_str)
    
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


if __name__ == "__main__":
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_data = []
    for i in tqdm(range(len(data)), desc="Processing users"):
        entry = data[i]
        payload = generate_payload(entry.get("keywords", []))
        name = entry.get("user", "unknown")
        try:
            response = extract_json_from_markdown(call_deepseek(payload))
        except Exception as e:
            print(f"Error processing user {entry.get('user', 'unknown')}: {e}")
        output_data.append({
            "user": name,
            "scores": response,
        })
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

