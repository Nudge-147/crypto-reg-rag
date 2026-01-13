#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_baseline_embed.py — 为 Baseline 数据构建向量索引
输入：baseline_data/cleaned
输出：baseline_data/indexes/faiss.index
"""

import json
import os
import time
from pathlib import Path

import faiss
import httpx
import numpy as np
from openai import OpenAI

try:
    from httpx_socks import SyncProxyTransport
except ImportError:
    SyncProxyTransport = None

# ====== 配置 ======
CLEANED_DIR = Path("baseline_data/cleaned")  # 读取无摘要数据
INDEX_DIR = Path("baseline_data/indexes")    # 存到独立文件夹
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.jsonl"

# 保持与 SAC 一致的模型配置，确保公平
EMBED_MODEL = "text-embedding-3-large"
BATCH_SIZE = 16
GPTS_BASE_URL = os.getenv("GPTS_BASE_URL", "https://api.gptsapi.net/v1")
PROXY_URL = os.getenv("PROXY_URL", "")


def create_client():
    api_key = os.getenv("GPTSAPI_API_KEY")
    if not api_key:
        raise ValueError("请设置 GPTSAPI_API_KEY")
    http_client = None
    if PROXY_URL and SyncProxyTransport:
        transport = SyncProxyTransport.from_url(PROXY_URL)
        http_client = httpx.Client(transport=transport)
    return OpenAI(api_key=api_key, base_url=GPTS_BASE_URL, http_client=http_client)


def embed_batch(texts, client: OpenAI):
    """调用嵌入接口，带简单重试。"""
    for _ in range(3):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return np.array([d.embedding for d in resp.data], dtype="float32")
        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(2)
    raise RuntimeError("Embedding failed after retries")


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = create_client()

    texts, metas = [], []
    files = sorted(CLEANED_DIR.glob("*.jsonl"))

    print(f"读取 Baseline 数据: {len(files)} 个文件")
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["augmented_text"])  # 这里是纯文本
                metas.append(obj)

    if not texts:
        print("未找到数据，请先运行 01_baseline_pipeline.py")
        return

    print(f"开始向量化 {len(texts)} 个切片 (Baseline)...")
    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        vecs = embed_batch(batch, client)
        all_vecs.append(vecs)
        print(f"   Processed {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    X = np.vstack(all_vecs)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # 归一化

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Baseline 索引构建完成，保存在: {INDEX_DIR}")


if __name__ == "__main__":
    main()
