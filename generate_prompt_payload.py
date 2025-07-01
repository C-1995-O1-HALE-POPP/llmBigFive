import json
import math
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def build_dataframe(raw_json: dict) -> pd.DataFrame:
    """把多层结构展开成行: [user, theme, theme_freq, token, token_freq]"""
    rows = []
    for user_obj in raw_json:
        user = user_obj["name"]
        for theme_block in user_obj["data"]:
            theme = theme_block["theme"]
            theme_freq = theme_block["frequency"] or 0
            for token, token_freq in (theme_block["keywords"] or {}).items():
                rows.append(
                    {
                        "user": user,
                        "theme": theme,
                        "theme_freq": theme_freq,
                        "token": token,
                        "token_freq": token_freq,
                    }
                )
    return pd.DataFrame(rows)

def compute_weights(df: pd.DataFrame) -> pd.DataFrame:
    # 计算每个 user-theme 的总词频
    theme_total = (
        df.groupby(["user", "theme"])["token_freq"]
        .transform("sum")
    )
    # 相对词频
    df["tf_rel"] = df["token_freq"] / theme_total
    # 权重
    df["weight"] = np.log(df["theme_freq"] + 1) * df["tf_rel"] * 100
    return df

def select_topk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """每名受试者保留权重最高的 k 条"""
    return (
        df.sort_values(["user", "weight"], ascending=[True, False])
        .groupby("user")
        .head(k)
        .reset_index(drop=True)
    )

def to_payload(df: pd.DataFrame) -> list[dict]:
    """转成 prompt payload 列表"""
    payload = []
    for user, sub in df.groupby("user"):
        payload.append(
            {
                "user": user,
                "keywords": [
                    {
                        "token": row["token"],
                        "weight": round(row["weight"], 4),
                        "context": row["theme"],
                    }
                    for _, row in sub.sort_values("weight", ascending=False).iterrows()
                ],
            }
        )
    return payload

def main(args):
    in_path = Path(args.input).expanduser()
    out_path = Path(args.output).expanduser()

    raw_data = json.loads(in_path.read_text(encoding="utf-8"))
    df = build_dataframe(raw_data)
    df = compute_weights(df)
    df = select_topk(df, args.topk)

    payload = to_payload(df)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Prompt payload saved to: {out_path} (users={len(payload)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 JSON 文件")
    parser.add_argument("--output", required=True, help="输出 payload JSON 文件")
    parser.add_argument("--topk", type=int, default=80, help="每人保留关键词数量")
    main(parser.parse_args())
