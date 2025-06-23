from pathlib import Path
import json, torch, pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "vladinc/bigfive-regression-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()                         # 关闭 dropout
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def build_essay(keywords: list[dict], use_weight=False) -> str:
    """把关键词列表转成一段输入文本"""
    segments = []
    for k in keywords:
        desc = k["description"]
        if use_weight:
            repeat = max(1, int(round(k["weight"] / 10)))   # 0~100 → 1~10 次
            segments.extend([desc] * repeat)
        else:
            segments.append(desc)
    return " ".join(segments)

def predict_big5(texts: list[str], batch_size: int = 8):
    """批量返回 Big-5 得分 (N, 5)"""
    all_outputs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.cpu()      # shape (B, 5)
        all_outputs.append(logits)
    return torch.cat(all_outputs, dim=0)               # (N, 5)

# 1) 读取数据
data_path = Path("users_keywords.json")
users = json.loads(data_path.read_text(encoding="utf-8"))

# 2) 构造文本
essays, user_ids = [], []
for u in users:
    essays.append(build_essay(u["keywords"], use_weight=False))
    user_ids.append(u["user"])

# 3) 推理
scores = predict_big5(essays, batch_size=8)            # Tensor (N,5)

# 4) 保存结果
traits = ["Openness", "Conscientiousness", "Extraversion",
          "Agreeableness", "Neuroticism"]
df = pd.DataFrame(scores.numpy(), columns=traits)
df.insert(0, "user", user_ids)

out_csv = "big5_results.csv"
df.to_csv(out_csv, index=False, float_format="%.4f")
print(f"Saved to {out_csv}")
