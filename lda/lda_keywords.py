import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

print("🔍 正在读取关键词数据...")

# 读取关键词列表
try:
    with open("keywords.json", "r", encoding="utf-8") as f:
        keywords = json.load(f)
    print(f"✅ 读取成功，共有 {len(keywords)} 个关键词。")
except Exception as e:
    print(f"❌ 读取失败：{e}")
    exit()

# 清洗关键词
keywords = [kw.strip() for kw in keywords if kw.strip()]
print(f"✅ 清洗完成，保留了 {len(keywords)} 个有效关键词。")

# 合并成文本数据
text_data = [" ".join(keywords)]
print("✅ 关键词合并为一段文本。")

# 向量化处理
print("🔄 正在进行文本向量化（CountVectorizer）...")
try:
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text_data)
    print(f"✅ 向量化完成，词汇总数：{len(vectorizer.get_feature_names_out())}")
except Exception as e:
    print(f"❌ 向量化失败：{e}")
    exit()

# LDA 模型
print("🔄 正在训练 LDA 模型...")
try:
    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_model.fit(X)
    print("✅ LDA 模型训练完成。")
except Exception as e:
    print(f"❌ 模型训练失败：{e}")
    exit()

# 获取主题关键词
print("📊 话题关键词提取中...")
try:
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda_model.components_):
        top_keywords = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        print(f"\n🧠 Topic {topic_idx + 1}: {', '.join(top_keywords)}")
    print("\n🎉 所有话题提取完成！")
except Exception as e:
    print(f"❌ 提取失败：{e}")
