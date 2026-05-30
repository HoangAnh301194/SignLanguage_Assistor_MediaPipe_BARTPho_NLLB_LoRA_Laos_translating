"""
Build or extend gloss-to-Vietnamese sentence dataset from VSL class_map.csv.

Usage:
    python data/build_dataset_from_class_map.py --class-map path/to/class_map.csv --output data/custom_sentence_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def slugify_gloss(text: str) -> str:
    text = text.strip().lower()
    replacements = {
        "à":"a","á":"a","ạ":"a","ả":"a","ã":"a","â":"a","ầ":"a","ấ":"a","ậ":"a","ẩ":"a","ẫ":"a","ă":"a","ằ":"a","ắ":"a","ặ":"a","ẳ":"a","ẵ":"a",
        "è":"e","é":"e","ẹ":"e","ẻ":"e","ẽ":"e","ê":"e","ề":"e","ế":"e","ệ":"e","ể":"e","ễ":"e",
        "ì":"i","í":"i","ị":"i","ỉ":"i","ĩ":"i",
        "ò":"o","ó":"o","ọ":"o","ỏ":"o","õ":"o","ô":"o","ồ":"o","ố":"o","ộ":"o","ổ":"o","ỗ":"o","ơ":"o","ờ":"o","ớ":"o","ợ":"o","ở":"o","ỡ":"o",
        "ù":"u","ú":"u","ụ":"u","ủ":"u","ũ":"u","ư":"u","ừ":"u","ứ":"u","ự":"u","ử":"u","ữ":"u",
        "ỳ":"y","ý":"y","ỵ":"y","ỷ":"y","ỹ":"y","đ":"d",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    return "_".join(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    class_map = Path(args.class_map)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(class_map, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss = row.get("gloss") or row.get("class_name") or ""
            gloss = gloss.strip()
            if not gloss:
                continue
            token = row.get("gloss_slug") or slugify_gloss(gloss)
            rows.append({"gloss_sequence": token, "vietnamese_sentence": gloss[0].upper() + gloss[1:] + "."})
            rows.append({"gloss_sequence": "toi muon " + token, "vietnamese_sentence": "Tôi muốn " + gloss.lower() + "."})
            rows.append({"gloss_sequence": "toi can " + token, "vietnamese_sentence": "Tôi cần " + gloss.lower() + "."})

    with open(output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gloss_sequence", "vietnamese_sentence"])
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", output)
    print("Rows:", len(rows))


if __name__ == "__main__":
    main()

