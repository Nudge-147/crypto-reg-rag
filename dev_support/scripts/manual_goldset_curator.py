#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_manual_curator.py — 人工辅助出题工具 (Human-in-the-Loop)
功能：
 1. 随机读取 cleaned_sac/ 下的法律切片。
 2. 屏幕展示：法域、文档名、SAC摘要、切片原文。
 3. 接收用户输入的问题 (Question)。
 4. 自动生成标准化的 JSON 测试条目 (包含绝对正确的 gold_sources)。
 5. 目标：快速构建 35 个高质量的黄金测试题 (Gold Standard)。
"""

import os
import json
import random
import glob

# ====== 配置 ======
SRC_DIR = "cleaned_sac"
OUTPUT_FILE = os.path.join("tests", "manual", "test_set_manual_35.json")
TARGET_COUNT = 35  # 目标题目数量

def load_random_chunk():
    """从所有 .jsonl 文件中随机抽取一个切片"""
    files = glob.glob(os.path.join(SRC_DIR, "*.jsonl"))
    if not files:
        print(f"❌ 错误: {SRC_DIR} 目录下没有找到 .jsonl 文件。请先运行 01 脚本。")
        return None, None

    # 随机选文件
    target_file = random.choice(files)
    doc_id = os.path.basename(target_file).replace(".jsonl", "")
    
    # 读取文件中的所有行
    chunks = []
    with open(target_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    if not chunks:
        return None, None

    # 随机选切片
    chunk = random.choice(chunks)
    return doc_id, chunk

def save_entry(entry, filepath):
    """追加保存到 JSON 文件"""
    data = []
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass # 文件损坏或为空，覆盖
    
    data.append(entry)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    print("\n" + "="*60)
    print("👨‍🏫  人工出题辅助系统 (Manual Curator)")
    print(f"目标: 构建 {TARGET_COUNT} 个黄金测试题")
    print("操作: 系统展示法条 -> 你输入问题 -> 自动保存")
    print("提示: 输入 's' 跳过当前段落, 'q' 保存退出")
    print("="*60 + "\n")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    current_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            try:
                current_count = len(json.load(f))
            except: pass

    while current_count < TARGET_COUNT:
        doc_id, chunk = load_random_chunk()
        if not chunk:
            continue

        # 提取展示信息
        jurisdiction = chunk.get('jurisdiction', 'UNKNOWN').upper()
        summary = chunk.get('summary', '无摘要')
        text = chunk.get('text', '') # 展示原始文本，阅读体验更好
        
        print(f"\n📚 进度: [{current_count + 1}/{TARGET_COUNT}] | 法域: {jurisdiction} | 文档: {doc_id}")
        print("-" * 60)
        print(f"📄【文档指纹/摘要】:\n{summary[:150]}...")
        print("-" * 30)
        print(f"📝【法律条文片段】:\n{text[:800]} ...") 
        print("-" * 60)

        # 获取用户输入
        question = input("👉 请输入基于此段落的问题 (s=跳过, q=退出): ").strip()

        if question.lower() == 'q':
            print("\n💾 进度已保存。再见！")
            break
        if question.lower() == 's' or question == "":
            print("⏭️  已跳过...")
            continue

        # 自动构建数据结构
        # 简单推断语言：如果法域是 CN 设为 zh，否则设为 en (可手动改)
        lang = "zh" if jurisdiction == "CN" else "en"
        
        new_entry = {
            "id": f"MANUAL_{current_count+1:02d}",
            "question_text": question,
            "question_language": lang, 
            "target_jurisdictions": [jurisdiction],
            "topic_category": "general", # 稍后可以手动细化
            "gold_sources": [
                {
                    "citation": doc_id, # 关键！直接使用文件名作为引用，DRM 绝对匹配
                    "jurisdiction": jurisdiction,
                    "relevance_note": "Human curated ground truth."
                }
            ],
            # 保留源切片信息方便复查
            "source_chunk_preview": text[:50] 
        }

        save_entry(new_entry, OUTPUT_FILE)
        current_count += 1
        print(f"✅ 第 {current_count} 题已录入！")

    if current_count >= TARGET_COUNT:
        print(f"\n🎉 恭喜！{TARGET_COUNT} 题目标已达成！文件保存在: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
