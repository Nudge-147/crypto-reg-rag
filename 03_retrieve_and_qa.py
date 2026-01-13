#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_retrieve_and_qa.py - 权威性感知检索与 RAG QA
功能：
 1. 定义权威性矩阵 (Authority Matrix)。
 2. 实现 score_chunk 多维打分函数 (相似度 + 权威性 + 语言奖励)。
 3. 执行重排序检索 (Re-ranking) 并生成问答。
"""

import os
import json
import time
import numpy as np
import faiss
from pathlib import Path
from typing import Optional, List, Dict
from openai import OpenAI
import openai
import httpx
try:
    from httpx_socks import SyncProxyTransport
except ImportError:
    SyncProxyTransport = None

# ====== 模型与配置 ======
PROXY_URL = os.getenv("PROXY_URL", "")
GPTS_BASE_URL = os.getenv("GPTS_BASE_URL", "https://api.gptsapi.net/v1")

INDEX_PATH = Path("indexes/faiss.index")
META_PATH = Path("indexes/meta.jsonl")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
QA_MODEL = os.getenv("QA_MODEL", "gpt-4o-mini") 
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_EMBED_RETRIES = 5


# >>>>>>>>>>>>>>>>>> 权威性数据模型 <<<<<<<<<<<<<<<<<<

class AuthorityMatrix:
    """定义不同法域在特定法律主题下的权威性权重 (0.0 - 1.0)"""
    def __init__(self):
        # 示例权威性矩阵：请根据您的研究需求定制
        self.matrix = {
            "data_protection": { # 示例：GDPR
                "EU": 1.0, 
                "US": 0.6,
                "SG": 0.7
            },
            "contract_law": { # 示例：英美判例法
                "US": 1.0,  
                "EU": 0.8,
                "SG": 0.9
            },
            "general": { # 默认 fallback
                "EU": 0.8, "US": 0.8, "SG": 0.8
            }
        }
    
    def get_score(self, topic: str, jurisdiction: str) -> float:
        topic_scores = self.matrix.get(topic, self.matrix["general"])
        # 如果法域不在主题列表中，给一个保守的分数 0.5
        return topic_scores.get(jurisdiction, 0.5) 

def classify_query_topics(query: str) -> List[str]:
    """[Abstract] 模拟 LLM 或关键词匹配进行主题分类"""
    if any(k in query.lower() for k in ["privacy", "gdpr", "data", "个人数据"]):
        return ["data_protection"]
    if any(k in query.lower() for k in ["contract", "agreement", "协议", "违约"]):
        return ["contract_law"]
    return ["general"]

# >>>>>>>>>>>>>>>>>> 初始化与加载 <<<<<<<<<<<<<<<<<<

def create_gptsapi_client(api_key: str, proxy_url: str = None):
    http_client = None
    if proxy_url and proxy_url.strip() != "" and SyncProxyTransport is not None:
        transport = SyncProxyTransport.from_url(proxy_url)
        http_client = httpx.Client(transport=transport, timeout=60.0)
        print(f"✅ 使用代理：{proxy_url}")
    
    client = OpenAI(api_key=api_key, http_client=http_client, base_url=GPTS_BASE_URL)
    return client, http_client

def load_index_and_meta(index_path: Path, meta_path: Path):
    if not index_path.exists(): raise FileNotFoundError(f"未找到向量索引：{index_path}")
    if not meta_path.exists(): raise FileNotFoundError(f"未找到元数据文件：{meta_path}")

    index = faiss.read_index(str(index_path))
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return index, metas

# >>>>>>>>>>>>>>>>>> 核心打分与检索逻辑 <<<<<<<<<<<<<<<<<<

def embed_query(query: str, client: OpenAI) -> np.ndarray:
    """将用户问题转换为 embedding 向量，并添加指数退避重试机制"""
    for attempt in range(1, MAX_EMBED_RETRIES + 1):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
            vec = np.array(resp.data[0].embedding, dtype="float32")
            vec = vec / (np.linalg.norm(vec) + 1e-12) 
            return vec.reshape(1, -1)

        except openai.RateLimitError:
            wait_time = 2 ** attempt
            print(f"⚠️ Rate Limit Error (429). 尝试 {attempt}/{MAX_EMBED_RETRIES}. 等待 {wait_time}s...")
            time.sleep(wait_time)
            continue

        except Exception as e:
            if attempt == MAX_EMBED_RETRIES:
                raise e
            print(f"❌ API 或网络错误. 尝试 {attempt}/{MAX_EMBED_RETRIES}. 等待 5s...")
            time.sleep(5)
            continue

    raise RuntimeError("Embedding failed after maximum retries.")


def score_chunk(sim_score: float, 
                chunk_meta: Dict, 
                authority_matrix: AuthorityMatrix, 
                query_topics: List[str], 
                query_lang: str,
                # 传入可调权重
                alpha: float,  # 语义相似度权重
                beta: float,   # 法域权威性权重
                gamma: float   # 语言匹配奖励权重
               ) -> float:
    """多维打分函数"""
    jurisdiction = chunk_meta.get("jurisdiction", "EU") 
    # 语言字段需要准确，这里简单设为 'en'，未来需实现语言检测
    chunk_lang = chunk_meta.get("language", "en") 
    
    # 1. 获取法域权威分
    auth_scores = [authority_matrix.get_score(t, jurisdiction) for t in query_topics]
    authority_val = max(auth_scores) if auth_scores else 0.5
    
    # 2. 语言匹配奖励 (Language Bonus)
    lang_bonus = 1.0 if chunk_lang == query_lang else 0.0
    
    # 3. 综合加权公式 (Authority-Aware Score)
    final_score = (alpha * sim_score) + (beta * authority_val) + (gamma * lang_bonus)
    
    return final_score


def retrieve_with_authority(query: str, 
                            client: OpenAI, 
                            index: faiss.IndexFlatIP, 
                            metas: list, 
                            top_k: int, 
                            authority_matrix: AuthorityMatrix,
                            candidate_k: int = 50, 
                            alpha: float = 0.6, 
                            beta: float = 0.3, 
                            gamma: float = 0.1):
    """
    权威性感知检索流程：检索 Top-N -> 重排序 -> 返回 Top-K。
    """
    query_vec = embed_query(query, client)

    # 2. 向量检索 (召回阶段): 检索 Top-N 个候选
    D, I = index.search(query_vec, candidate_k) 
    
    # 3. 解析查询主题和语言
    query_topics = classify_query_topics(query)
    query_lang = "en" # 简化：假设查询是英文

    scored_candidates = []
    # 4. 重排序阶段
    for rank, idx in enumerate(I[0]):
        sim_score = float(D[0][rank]) 
        chunk_meta = metas[idx]      
        
        final_score = score_chunk(
            sim_score, chunk_meta, authority_matrix, query_topics, query_lang,
            alpha=alpha, beta=beta, gamma=gamma
        )
        
        # 准备输出结果
        result_item = {
            "final_score": final_score,
            "original_sim": sim_score,
            "chunk_meta": chunk_meta # 包含 jurisdiction, augmented_text, summary 等
        }
        scored_candidates.append(result_item)
            
    # 5. 按最终得分排序并返回 Top-K
    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return scored_candidates[:top_k]


def generate_answer(question: str, retrieved_context: list, client: OpenAI):
    """用 GPTsAPI 的对话模型生成法规问答"""
    # 确保 LLM 拿到的是原始文本 (text) 而不是 Augmented Text
    context_text = "\n\n".join([f"- {item}" for item in retrieved_context])

    prompt = (
        "你是一名国际加密资产合规顾问，请基于下列法规原文回答问题。\n"
        "务必引用相关条文编号或标题，突出法域。\n\n"
        f"【法规条文摘要】:\n{context_text}\n\n"
        f"【用户问题】:\n{question}\n\n"
        "请在回答末尾以『出处：[法域] 文档名』格式标明引用的主要来源。"
    )

    resp = client.chat.completions.create(
        model=QA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()

# ====== 主程序 ======
def main():
    api_key = os.getenv("GPTSAPI_API_KEY")
    if not api_key: raise RuntimeError("请先设置环境变量 GPTSAPI_API_KEY")

    client, http_client = create_gptsapi_client(api_key, PROXY_URL)
    index, metas = load_index_and_meta(INDEX_PATH, META_PATH)
    print(f"✅ Loaded FAISS index ({index.ntotal} vectors)")
    authority_matrix = AuthorityMatrix()

    while True:
        question = input("\n请输入您的法规问题（输入 exit 退出）：\n> ").strip()
        if not question or question.lower() in ["exit", "quit"]: break

        try:
            top_k_results = retrieve_with_authority(
                question, client, index, metas, TOP_K, authority_matrix
            )
            
            # 提取 LLM 需要的原始文本和法域信息
            context_for_llm = []
            for r in top_k_results:
                 meta = r['chunk_meta']
                 # LLM QA 只需要原始文本 (text)
                 context_for_llm.append(f"[{meta.get('jurisdiction')}/{meta.get('doc_id')}] - {meta.get('text')}") 

            # 2️⃣ 生成回答
            answer = generate_answer(question, context_for_llm, client)
            
        except Exception as e:
            print(f"❌ 无法解析或生成回答，请检查网络/代理/密钥配置。详细信息：{e}")
            continue

        # 4️⃣ 结构化输出
        print("\n=== 💬 回答 ===")
        print(answer)
        
        print("\n=== 📄 重排序结果 (Top 5) ===")
        for i, r in enumerate(top_k_results, start=1):
            meta = r['chunk_meta']
            print(f"  {i}. Final={r['final_score']:.4f} | Sim={r['original_sim']:.4f} | Juri={meta.get('jurisdiction')} | Doc={meta.get('doc_id')} | {meta.get('text')[:40]}...")

    if http_client:
        http_client.close()

if __name__ == "__main__":
    main()
