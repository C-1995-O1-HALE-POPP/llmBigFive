#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_users_json.py

功能：
1. 从 Excel（xlsx）文件读取“用户名称 ↔ 编号”映射
2. 根据映射修改 JSON：
   a. 若 JSON 中的 name 在映射中不存在，则删除该条目
   b. 若存在，则在条目中新增字段 "uid"，其值为映射中的编号

用法（命令行示例）：
    python update_users_json.py \
        --excel "人格朋友圈尝试代入分析.xlsx" \
        --json  "人格特质50位受试者原始数据.json" \
        --output "updated_人格特质50位受试者.json"
"""

import json
import argparse
from pathlib import Path

import pandas as pd

# ------------------ 核心函数 ------------------ #
def build_uid_map(excel_path: Path,
                  sheet_name: str | int | None = 0,
                  name_col: str = "name",
                  uid_col: str = "uid") -> dict[str, int]:
    """
    读取 Excel，构造 {name: uid} 映射。
    - sheet_name 可指定工作表名称或索引，默认读第一个工作表
    - name_col / uid_col 分别是“姓名列”和“编号列”列名
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, dtype={uid_col: str})
    if name_col not in df.columns or uid_col not in df.columns:
        raise ValueError(
            f"Excel 缺少列：{name_col} 或 {uid_col}，请检查列名或通过参数指定"
        )
    # 去掉空值，strip 首尾空格
    df[name_col] = df[name_col].astype(str).str.strip()
    df[uid_col] = df[uid_col].astype(str).str.strip()
    # 去重：若同一 name 出现多次，保留首行并发出警告
    if df[name_col].duplicated().any():
        dup_names = df.loc[df[name_col].duplicated(), name_col].unique()
        print(f"[WARN] Excel 中以下姓名出现重复，仅使用首次出现的编号: {dup_names}")
        df = df.drop_duplicates(subset=name_col, keep="first")
    return df.set_index(name_col)[uid_col].to_dict()

def update_json(json_path: Path, uid_map: dict[str, int]) -> list[dict]:
    """
    按 uid_map 过滤并更新 JSON：
    - 不在 uid_map 中的条目被移除
    - 匹配到的条目添加 "uid" 字段
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    new_data = []
    for entry in data:
        name = str(entry.get("name", entry.get("user", ""))).strip()
        if name in uid_map:
            entry["uid"] = uid_map[name]
            new_data.append(entry)
    return new_data

# ------------------ 命令行接口 ------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据 Excel 更新 JSON 用户信息")
    parser.add_argument("--excel", required=True, type=Path, help="包含姓名与编号的 xlsx 文件")
    parser.add_argument("--json",  required=True, type=Path, help="原始 JSON 文件")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出 JSON（默认覆盖原文件）")
    parser.add_argument("--sheet", default=0,
                        help="Excel 工作表名称或索引（默认 0，即第一个工作表）")
    parser.add_argument("--name-col", default="name",
                        help="Excel 中的姓名列名（默认 'name'）")
    parser.add_argument("--uid-col", default="uid",
                        help="Excel 中的编号列名（默认 'uid'）")
    args = parser.parse_args()

    uid_map = build_uid_map(args.excel,
                            sheet_name=args.sheet,
                            name_col=args.name_col,
                            uid_col=args.uid_col)
    updated = update_json(args.json, uid_map)

    out_path = args.output or args.json  # 覆盖或另存为
    out_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] 更新完成，写入 {out_path}，共保留 {len(updated)} 条记录")
