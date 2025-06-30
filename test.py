
import json
import argparse
from pathlib import Path
from typing import Iterable, Set, Dict, Any


def gather_keywords(records: Iterable[Dict[str, Any]]) -> Set[str]:
    """
    遍历最外层记录，深挖 generated → <image_id> → keywords → '关键词列表'，
    将其中的关键词加入集合去重。
    """
    unique: Set[str] = set()

    for item in records:
        for img in item.get("generated", {}).values():
            kw_list = (
                img.get("keywords", {})
                .get("关键词列表", [])
            )
            unique.update(kw_list)

    return unique


def main():
    print("Counting unique keywords in '关键词列表'...\n")
    parser = argparse.ArgumentParser(
        description="Count unique keywords in '关键词列表'."
    )
    parser.add_argument(
        "json_file",
        help="Path to the JSON file exported from your dataset."
    )
    args = parser.parse_args()

    data = json.loads(Path(args.json_file).read_text(encoding="utf-8"))

    keywords = gather_keywords(data)
    print(f"Unique keyword count: {len(keywords)}\n")

    # 按字典序输出，方便查阅
    with open("keywords_with_user.json", "w", encoding="utf-8") as f:
        json.dump(sorted(keywords), f, ensure_ascii=False, indent=2)
main()
