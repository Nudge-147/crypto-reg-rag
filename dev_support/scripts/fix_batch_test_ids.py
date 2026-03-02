import glob
import json
import os
import re
from typing import Optional


def infer_batch_num_from_filename(filename: str) -> Optional[int]:
    """从文件名中提取批次号（B01 -> 1）。未匹配到则返回 None。"""
    match = re.search(r"B(\d{2})", filename)
    return int(match.group(1)) if match else None


def prefix_test_set_ids(filename: str, batch_num: Optional[int] = None):
    """
    读取 JSON 文件，将数值 ID 转换为唯一的 BXX_QYY 格式，并清理 LLM 生成的占位符。

    Args:
        filename: 原始测试集文件名 (如 tests/test_set_B01.json)。
        batch_num: 当前批次的编号 (如 1)。若为 None，则尝试从文件名推断。
    """
    if not os.path.exists(filename):
        print(f"❌ Error: File {filename} not found. Please ensure your original JSON is saved.")
        return None

    # 自动推断批次号（若未指定）
    if batch_num is None:
        batch_num = infer_batch_num_from_filename(filename)
        if batch_num is None:
            print(f"❌ Error: Cannot infer batch number from {filename}. Please pass batch_num explicitly.")
            return None

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ Error: File {filename} contains invalid JSON.")
        return None

    batch_prefix = f"B{batch_num:02d}"
    
    for idx, item in enumerate(data):
        # 使用旧 ID 或列表索引作为数字部分
        old_numeric_id = item.get("id", idx + 1)
        
        # 1. 格式化 ID: B01_Q01, B01_Q02, ...
        new_id = f"{batch_prefix}_Q{idx + 1:02d}"
        
        # 备份和替换 ID
        item["original_id"] = old_numeric_id
        item["id"] = new_id
        
        # 2. 清理 GPT 在 gold_spans 中添加的占位符
        if 'gold_spans' in item:
            for span in item['gold_spans']:
                if 'excerpt' in span:
                    # 移除所有 :contentReference[...] 及其后的内容 (这是 GPT 倾向于添加的占位符)
                    span['excerpt'] = re.split(r':contentReference\[.*?\]', span['excerpt'])[0].strip()

    # 将修复后的数据保存为新的文件
    out_dir = os.path.dirname(filename) or "."
    new_filename = os.path.join(out_dir, f"test_set_{batch_prefix}_fixed.json")
    with open(new_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ ID 修复完成。新文件已保存为 {new_filename}。")
    return new_filename


def process_all_test_sets(pattern: str = "dev_support/tests/batches_legacy/test_set_B[0-9][0-9].json"):
    """
    处理匹配模式下的所有测试集文件
    （如 dev_support/tests/batches_legacy/test_set_B01.json, ...）。
    """
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"❌ No files found matching pattern: {pattern}")
        return []

    results = []
    for file in files:
        result = prefix_test_set_ids(filename=file)
        if result:
            results.append(result)
    return results


if __name__ == "__main__":
    # 默认处理 dev_support/tests/batches_legacy/test_set_B01/02/03... 等文件
    process_all_test_sets()
