#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：读取 cleaned_sac/ 下的 SAC 法规 chunk 数据 → 用 GPTsAPI 生成向量 → 构建 FAISS 索引  
使用说明：
 1. 确保已安装所有依赖 (pip install openai httpx httpx-socks faiss-cpu numpy tqdm tiktoken)
 2. export GPTSAPI_API_KEY="你的 GPTsAPI 密钥"
 3. 在项目根目录运行： python3 02_embed_build_openai.py
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from tqdm import tqdm

from openai import OpenAI
import openai
import httpx
# 只有设置了 PROXY_URL 环境变量才需要 httpx_socks
try:
    from httpx_socks import SyncProxyTransport
except ImportError:
    SyncProxyTransport = None
    
import tiktoken  # 用于 token 估算

# ====== 配置参数 ======
# 确保指向 SAC 流程的输出目录
CLEANED_DIR = Path("cleaned_sac") 

INDEX_DIR = Path("indexes")
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.jsonl"

PROXY_URL = os.getenv("PROXY_URL", "") # 留空则不使用代理

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TIMEOUT_SECONDS = float(os.getenv("TIMEOUT_SECONDS", "60.0"))
DELAY_BETWEEN_BATCHES = float(os.getenv("DELAY_BETWEEN_BATCHES", "2.0"))

# GPTsAPI Base URL
GPTS_BASE_URL = os.getenv("GPTS_BASE_URL", "https://api.gptsapi.net/v1")

def create_client_with_proxy(proxy_url: Optional[str], api_key: str):
    """
    创建客户端，如果 PROXY_URL 设置有效则使用代理。
    """
    http_client = None
    if proxy_url and proxy_url.strip().lower() not in {"", "none", "null", "no"}:
        if SyncProxyTransport is None:
            print("⚠️ 缺少 httpx-socks 库，无法使用代理。请运行: pip install httpx-socks")
        else:
            try:
                transport = SyncProxyTransport.from_url(proxy_url)
                http_client = httpx.Client(transport=transport, timeout=TIMEOUT_SECONDS)
                print(f"🔌 使用代理: {proxy_url}")
            except Exception as e:
                print(f"⚠️ 代理配置错误 ({e}), 回退到直连.")
                http_client = None

    client = OpenAI(api_key=api_key, http_client=http_client, base_url=GPTS_BASE_URL)
    print(f"🧩 Base URL being used: {GPTS_BASE_URL}")
    return client, http_client


def embed_batch(texts: list[str], client: OpenAI) -> np.ndarray:
    # token 裁剪
    encoder = tiktoken.get_encoding("cl100k_base")
    max_tokens = 8000
    processed = []
    for txt in texts:
        ids = encoder.encode(txt)
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
            txt = encoder.decode(ids)
        processed.append(txt)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=processed,
                timeout=TIMEOUT_SECONDS
            )
            vecs = [item.embedding for item in resp.data]
            return np.array(vecs, dtype="float32")

        except openai.RateLimitError as e:
            retry_after = getattr(e, "retry_after", None)
            wait = retry_after if retry_after is not None else (2 ** attempt)
            print(f"⚠️ RateLimitError attempt {attempt}/{MAX_RETRIES}: {e}")
            print(f"   Retrying after {wait}s …")
            time.sleep(wait)
            continue
        # ... (省略其他错误捕获以保持简洁)
        except Exception as e:
            print(f"❌ 遇到错误，尝试次数 {attempt}/{MAX_RETRIES}: {e}")
            if attempt == MAX_RETRIES:
                 raise
            time.sleep(5)
            continue

    raise RuntimeError(f"Embedding failed after {MAX_RETRIES} attempts")

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def load_chunks(cleaned_dir: Path):
    texts = []
    metas = []
    jsonl_files = sorted(cleaned_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise RuntimeError(f"No .jsonl files found in {cleaned_dir}")
    for jf in jsonl_files:
        print(f"🔍 Loading: {jf}")
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # 关键修改：嵌入 augmented_text (摘要+正文)
                    txt = obj.get("augmented_text", "").strip() 
                    if not txt:
                        continue
                    texts.append(txt)
                    # 关键修改：将完整的 chunk 对象作为元数据存储，以便检索时访问所有 SAC 信息
                    metas.append(obj) 
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping invalid JSON line in {jf}: {e}")
                    continue
    print(f"✅ Loaded {len(texts)} SAC chunks")
    return texts, metas

def main():
    api_key = os.getenv("GPTSAPI_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 GPTSAPI_API_KEY")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    client, http_client = create_client_with_proxy(PROXY_URL, api_key)

    try:
        texts, metas = load_chunks(CLEANED_DIR)
        if not texts:
            raise RuntimeError("No valid SAC text chunks found! 请确保 01_sac_pipeline.py 已经成功运行。")

        total = len(texts)
        batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        all_vecs = []
        print(f"🚀 Starting embedding generation for {total} chunks...")
        for idx, i in enumerate(range(0, total, BATCH_SIZE)):
            batch = texts[i : i + BATCH_SIZE]
            print(f"➡️ Processing batch {idx+1}/{batches}, items {i+1}-{i+len(batch)} …")
            vecs = embed_batch(batch, client)
            all_vecs.append(vecs)
            print(f"⏱ Sleeping {DELAY_BETWEEN_BATCHES}s between batches …")
            time.sleep(DELAY_BETWEEN_BATCHES)

        X = np.vstack(all_vecs)
        print(f"✅ Embedding matrix shape: {X.shape}")

        X = l2_normalize(X)

        dim = X.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(X)
        faiss.write_index(index, str(INDEX_PATH))
        print(f"💾 Saved index to: {INDEX_PATH}")

        with open(META_PATH, "w", encoding="utf-8") as f_out:
            for meta in metas:
                f_out.write(json.dumps(meta, ensure_ascii=False) + "\n")
        print(f"💾 Saved metadata to: {META_PATH}")

        print("🎉 All done! SAC Embedding + index built successfully.")
    finally:
        if http_client:
             http_client.close()

if __name__ == "__main__":
    main()