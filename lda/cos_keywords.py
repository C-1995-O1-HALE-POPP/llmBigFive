import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

# ---------------- 参数 ----------------
DATA       = Path("keywords_副本.json")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLD  = 0.30             # 距离阈值：余弦相似 ≥ 0.70
OUT_FILE   = Path("clusters.json")

# ---------------- 日志 ----------------
logger.add("cluster.log",
           rotation="10 MB",
           retention="7 days",
           compression="zip",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# ---------------- 读文件 & 清洗 ----------------
words   = json.loads(DATA.read_text("utf-8"))
cleaned = {w.strip() for w in words if w and isinstance(w, str) and w.strip()}

re_date = re.compile(r"\d{4}年|\d{1,2}[/-]\d{1,2}")
re_num  = re.compile(r"^\d+([.:kmKM]+)?$")

semantic_words = [w for w in cleaned
                  if not (re_num.match(w) or re_date.search(w))]

logger.info(f"语义词数量: {len(semantic_words)}")

# ---------------- 向量化 ----------------
model = SentenceTransformer(MODEL_NAME)
emb   = model.encode(semantic_words, show_progress_bar=True).astype(np.float32)
logger.info(f"向量化完成: {emb.shape}")

k = 10   # 示例，实际可用 pyclustering.elbow_method 或 yellowbrick

kmeans = MiniBatchKMeans(
    n_clusters=k,
    init="k-means++",
    batch_size=2048,
    max_iter=300,
    n_init=20,          # 多次初始化挑最优
    random_state=42
)
labels = kmeans.fit_predict(emb)

# ---------- 导出 ----------
clusters = {i: [] for i in range(10)}
for w, lb in zip(words, labels):
    clusters[int(lb)].append(w)

OUT_FILE.write_text(
    json.dumps({k: sorted(v) for k, v in clusters.items()},
               ensure_ascii=False, indent=2),
    "utf-8")
print(f"✅ 已写入 {OUT_FILE.resolve()}")