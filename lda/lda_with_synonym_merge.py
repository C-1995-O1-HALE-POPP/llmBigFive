#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lda_with_synonym_merge.py
------------------------------------------
Step 0 : 读取原始关键词列表 (.json)
Step 1 : 基础规范化（Unicode NFKC、繁→简、大小写、空白折叠、词形化）
Step 2 : 句向量 + AgglomerativeClustering 做同义词/近义词归并
Step 3 : Canonical Map 替换得到“规范词”序列
Step 4 : CountVectorizer 构建词袋  ➜  Latent Dirichlet Allocation
Step 5 : 输出
    - console：每个主题的 top-15 关键词
    - files  ：
        * canonical_map.json      原词 → 规范词
        * lda_topics.json         {topic_i: [kw1, kw2, …]}
        * lda_doc_labels.csv      每行：原词, 规范词, topic_id
"""

import argparse, json, re, unicodedata, csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from opencc import OpenCC
import nltk
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ---------------------------- CLI 参数 ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input",     default="keywords_副本.json",
                    help="原始关键词 JSON 列表文件")
parser.add_argument("--n_topics",  type=int, default=5,
                    help="LDA 主题数")
parser.add_argument("--dist_th",   type=float, default=0.30,
                    help="余弦距离阈值 (0.30 ≈ 相似度 ≥ 0.70)")
parser.add_argument("--topn",      type=int, default=15,
                    help="每个主题展示前 N 个关键词")
args = parser.parse_args()

print(f"▶︎ loading {args.input}")
keywords_raw = json.loads(Path(args.input).read_text("utf-8"))
print(f"  total keywords: {len(keywords_raw):,}")

# ----------------------- 1. 基础规范化 ----------------------------
lemmatizer = WordNetLemmatizer()
cc = OpenCC("t2s")

def normalize(token: str) -> str:
    token = unicodedata.normalize("NFKC", token)
    token = cc.convert(token)               # 繁体→简体
    token = token.lower().strip()
    token = re.sub(r"\s+", " ", token)      # 折叠多空格
    token = lemmatizer.lemmatize(token)     # 英文词形归一
    return token

keywords_norm = [normalize(k) for k in keywords_raw if k.strip()]
keywords_norm = list(dict.fromkeys(keywords_norm))  # 保序去重
print(f"  after normalize & dedup: {len(keywords_norm):,}")

# ----------------------- 2. 同义词聚合 ----------------------------
print("▶︎ encoding sentence embeddings …")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
emb = model.encode(
    keywords_norm, batch_size=512,
    show_progress_bar=True, normalize_embeddings=True
)

print("▶︎ agglomerative clustering …")
cluster = AgglomerativeClustering(
    n_clusters=None, linkage="average",
    metric="cosine", distance_threshold=args.dist_th
).fit(emb)

cluster2words = defaultdict(list)
for w, lab in zip(keywords_norm, cluster.labels_):
    cluster2words[lab].append(w)

def pick_canonical(ws):
    # 最短优先，其次字典序
    return sorted(ws, key=lambda s: (len(s), s))[0]

canonical_map = {w: pick_canonical(cluster2words[lab])
                 for lab, ws in cluster2words.items() for w in ws}
print(f"  clusters found: {len(cluster2words):,}")

# 保存映射
Path("canonical_map.json").write_text(
    json.dumps(canonical_map, ensure_ascii=False, indent=2), "utf-8"
)

# ----------------------- 3. 替换为规范词 ---------------------------
keywords_canonical = [canonical_map[w] for w in keywords_norm]

# ----------------------- 4. LDA 训练 ------------------------------
print("▶︎ vectorizing with CountVectorizer …")
vectorizer = CountVectorizer(
    token_pattern=r'[\u4e00-\u9fa5]+|[a-z]+',      # 中英文 token
    stop_words=None,                               # 如需停用词可自行添加
    max_features=5000
)
X = vectorizer.fit_transform(keywords_canonical)

print(f"▶︎ training LDA (n_topics={args.n_topics}) …")
lda = LatentDirichletAllocation(
    n_components=args.n_topics,
    learning_method="online",
    max_iter=20,
    random_state=42
).fit(X)

vocab = vectorizer.get_feature_names_out()
topic_keywords = defaultdict(list)

print("\n========== LDA TOPICS ==========")
for idx, comp in enumerate(lda.components_):
    top_idx = comp.argsort()[-args.topn:][::-1]
    words = [vocab[i] for i in top_idx]
    topic_keywords[f"topic_{idx+1}"] = words
    print(f"Topic {idx+1}: {', '.join(words)}")

# -------------------- 5. 文档 → 主题 标签 -------------------------
print("\n▶︎ assigning each keyword to its best-probability topic …")
doc_topic = lda.transform(X).argmax(axis=1)

# CSV：原词, 规范词, 主题号
with open("lda_doc_labels.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["raw_word", "canonical_word", "topic_id"])
    for raw, canon, label in zip(keywords_norm, keywords_canonical, doc_topic):
        writer.writerow([raw, canon, label+1])

# JSON：{topic_i: [kw1, kw2, …]}
Path("lda_topics.json").write_text(
    json.dumps(topic_keywords, ensure_ascii=False, indent=2), "utf-8"
)

print("\n✅ Done.")
print("  • canonical_map.json    归并映射")
print("  • lda_topics.json       每个主题的 Top 关键词")
print("  • lda_doc_labels.csv    每个词对应的主题标签")
