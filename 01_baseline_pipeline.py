#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_baseline_pipeline.py — Naive RAG (No Summary)
功能：
 1. 读取 PDF 全文
 2. 直接进行物理切片 (Chunking)
 3. 不生成摘要，augmented_text = 纯原文
 4. 保存到 baseline_data/cleaned
"""

import json
import re
from pathlib import Path
import fitz  # PyMuPDF

# ====== 配置 ======
RAW_ROOT = Path("raw")
# 输出到新文件夹，与 SAC 区分开
OUTPUT_DIR = Path("baseline_data/cleaned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 包含所有法域
JURISDICTIONS = ["eu", "us", "sg", "cn", "br", "sv", "jp", "uk", "hk", "kr", "ch", "uae"]

CHUNK_SIZE = 500
CHUNK_OVERLAP = 0


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and soft hyphens from PDF extraction."""
    if not text:
        return ""
    text = text.replace("\u00A0", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_full_text(pdf_path: Path) -> str:
    """Extract raw text from PDF."""
    doc = fitz.open(str(pdf_path))
    full_text = []
    for page in doc:
        full_text.append(page.get_text())
    return normalize_whitespace("\n".join(full_text))


def recursive_chunk_text(text: str, chunk_size: int = 500, overlap: int = 0):
    """Chunk text with simple newline-aware splitting."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            lookback = text.rfind("\n", start, end)
            if lookback != -1 and lookback > start + chunk_size * 0.8:
                end = lookback + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def process_file_baseline(pdf_path: Path):
    doc_id = pdf_path.stem
    jurisdiction = pdf_path.parent.name.upper()

    # 1. 提取全文
    full_text = extract_full_text(pdf_path)
    if len(full_text) < 100:
        print(f"Skip {doc_id} (content too short)")
        return

    # 2. 切分 (无摘要步骤)
    raw_chunks = recursive_chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

    # 3. 写入
    out_file = OUTPUT_DIR / f"{doc_id}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for idx, chunk_body in enumerate(raw_chunks):
            # 关键对照点：没有 Summary
            record = {
                "chunk_id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "jurisdiction": jurisdiction,
                "text": chunk_body,
                "augmented_text": chunk_body,  # 只有原文，没有摘要
                "summary": "",
                "chunk_index": idx,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Baseline processed: {doc_id}")


def main():
    print("Start Baseline (Naive RAG) pipeline...")
    for jur in JURISDICTIONS:
        folder = RAW_ROOT / jur
        if not folder.exists():
            continue

        pdfs = sorted(folder.glob("*.pdf"))
        for pdf in pdfs:
            process_file_baseline(pdf)


if __name__ == "__main__":
    main()
