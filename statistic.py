import json

# 1. 读入数据 ---------------------------------------------------------
with open("人格特质图像识别+分类.json", "r", encoding="utf-8") as f:
    moments = json.load(f)          # moments 是一个列表，每个元素是一条朋友圈记录

# 2. 计算指标 ---------------------------------------------------------
users      = {item["user"] for item in moments}          # 不重复用户集合
num_users  = len(users)                                  # 用户数
num_posts  = len(moments)                                # 朋友圈条目数
num_images = sum(len(item.get("images", [])) for item in moments)   # 图片总数

# 3. 打印结果 ---------------------------------------------------------
print(f"不重复用户数: {num_users}")
print(f"朋友圈条目总数: {num_posts}")
print(f"图片总数: {num_images}")
