#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_sac_pipeline.py — 实现 SAC (Summary-Augmented Chunking)
功能：
 1. 读取 PDF 全文
 2. 调用 LLM 生成文档级摘要 (Document Fingerprint)
 3. 对全文进行切片 (Chunking)
 4. 拼接: Augmented Text = Summary + "\n\n" + Chunk Body
 5. 保存结果用于嵌入
"""

import os
import json
import time
import re
from pathlib import Path
import fitz  # PyMuPDF
from openai import OpenAI
import httpx
from httpx_socks import SyncProxyTransport

# ====== 配置 ======
RAW_ROOT = Path("raw")
OUTPUT_DIR = Path("cleaned_sac")  # 新的输出目录，区分于普通的 cleaned
OUTPUT_DIR.mkdir(exist_ok=True)

JURISDICTIONS = ["eu", "us", "sg", "cn", "br", "sv", "jp", "uk", "hk", "kr", "ch", "uae"]

# [cite_start]SAC 参数 [cite: 160-165, 360-380]
SUMMARY_CHAR_LIMIT = 150  # 论文建议摘要长度
CHUNK_SIZE = 500          # 切片大小
CHUNK_OVERLAP = 0         # 论文中设置为无重叠

# LLM 配置
GPTS_BASE_URL = os.getenv("GPTS_BASE_URL", "https://api.gptsapi.net/v1")
API_KEY = os.getenv("GPTSAPI_API_KEY")
PROXY_URL = os.getenv("PROXY_URL", "") # 留空则不使用代理
SUMMARY_MODEL = "gpt-4o-mini" # 用便宜的模型生成摘要即可

# ====== 工具函数 ======

def create_client():
    if not API_KEY:
        raise ValueError("❌ 缺少 GPTSAPI_API_KEY 环境变量！")
    
    if PROXY_URL and PROXY_URL.strip() != "":
        print(f"🔌 使用代理: {PROXY_URL}")
        transport = SyncProxyTransport.from_url(PROXY_URL)
        http_client = httpx.Client(transport=transport, timeout=60.0)
        return OpenAI(api_key=API_KEY, base_url=GPTS_BASE_URL, http_client=http_client)
    else:
        return OpenAI(api_key=API_KEY, base_url=GPTS_BASE_URL)

def normalize_whitespace(text: str) -> str:
    if not text: return ""
    text = text.replace("\u00A0", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_full_text(pdf_path: Path) -> str:
    """提取整个 PDF 的文本用于总结"""
    doc = fitz.open(str(pdf_path))
    full_text = []
    # 为了省钱/省时间，如果文件巨大（超过50页），只取前10页和后5页生成摘要
    # 这里为了演示，我们提取全文（因为法律文件通常需要通读）
    for page in doc:
        full_text.append(page.get_text())
    return normalize_whitespace("\n".join(full_text))

def generate_document_summary(client, text: str, doc_name: str) -> str:
    """[Goal 1] 调用 LLM 生成文档摘要 (Document Fingerprint)"""
    print(f"🤖 正在为 {doc_name} 生成摘要...")
    
    # 截断过长的文本防止 Token 溢出 (例如只取前 15000 字符用于摘要)
    # 实际生产中可以使用 Map-Reduce 摘要，但这里简化处理
    context_text = text[:15000] 

    prompt = (
        f"你是一个法律专家。请为以下法律文档生成一个极简的‘文档指纹’（摘要）。\n"
        f"要求：\n"
        f"1. 包含核心法律主题、适用范围和关键实体。\n"
        f"2. 长度严格控制在 {SUMMARY_CHAR_LIMIT} 个字符左右。\n"
        f"3. 不要废话，直接输出摘要内容。\n\n"
        f"文档内容摘要（截取）：\n{context_text}..."
    )

    try:
        resp = client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        summary = resp.choices[0].message.content.strip()
        print(f"✅ 摘要生成: {summary}")
        return summary
    except Exception as e:
        print(f"⚠️ 摘要生成失败: {e}，将使用文件名代替。")
        return f"Document: {doc_name}"

def recursive_chunk_text(text: str, chunk_size=500, overlap=0):
    """简单的切片逻辑"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        # 简单优化：尝试在换行或句号处截断
        if end < text_len:
            lookback = text.rfind('\n', start, end)
            if lookback != -1 and lookback > start + chunk_size * 0.8:
                end = lookback + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    return chunks

# ====== 主流程 ======

def process_file_sac(client, pdf_path: Path):
    doc_id = pdf_path.stem
    jurisdiction = pdf_path.parent.name.upper()
    
    # 1. 提取全文
    full_text = extract_full_text(pdf_path)
    if len(full_text) < 100:
        print(f"⏭️ 跳过 {doc_id} (内容太少，可能是扫描件)")
        return

    # 2. 生成摘要 (SAC 核心步骤)
    doc_summary = generate_document_summary(client, full_text, doc_id)

    # 3. 切分
    raw_chunks = recursive_chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

    # 4. 构建 SAC 数据并写入
    out_file = OUTPUT_DIR / f"{doc_id}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for idx, chunk_body in enumerate(raw_chunks):
            # [SAC 核心] 拼接摘要 + 原始内容
            augmented_text = f"Doc Summary: {doc_summary}\n\nContent: {chunk_body}"
            
            record = {
                "chunk_id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "jurisdiction": jurisdiction,
                "text": chunk_body,         # 原始文本 (用于显示)
                "augmented_text": augmented_text, # 用于嵌入向量的文本！
                "summary": doc_summary,     # 存储摘要以备查
                "chunk_index": idx
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"💾 已保存 {len(raw_chunks)} 个 SAC 切片到 {out_file}")

def main():
    print("🚀 启动 SAC (Summary-Augmented Chunking) 流水线...")
    client = create_client()
    
    for jur in JURISDICTIONS:
        folder = RAW_ROOT / jur
        if not folder.exists(): continue
        
        pdfs = sorted(folder.glob("*.pdf"))
        print(f"\n📂 处理法域: {jur} ({len(pdfs)} 文件)")
        
        for pdf in pdfs:
            process_file_sac(client, pdf)
            time.sleep(1) # 避免 API 速率限制

if __name__ == "__main__":
    main()
