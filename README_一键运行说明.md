# 最终版一键运行说明（Crypto Regulation RAG）

本说明用于配合项目报告提交，提供可复现的运行步骤（API 启动、接口回归、SAC 与 Baseline 对照评测）。

## 一、项目内容（提交包建议包含）

- 项目报告（PDF）
- 源代码（本项目目录）
- 本说明文件：`README_一键运行说明.md`

## 二、核心程序文件（按流程理解）

说明：为保证复现命令稳定，项目保留了编号脚本命名（`01~06`）。可按下表理解其功能。

### 主系统（SAC + 多法域检索）

- `01_sac_pipeline.py`
  - 作用：SAC 预处理（PDF -> 摘要增强切片）
- `02_embed_build_openai.py`
  - 作用：SAC 向量化并构建主索引（`indexes/`）
- `03_retrieve_and_qa.py`
  - 作用：检索与重排核心（法域过滤、多法域分桶召回、权威性重排、问答）
- `06_api_server.py`
  - 作用：HTTP API 封装（`/jurisdictions`, `/query`）

### 对照系统（Baseline / 非SAC）

- `01_baseline_pipeline.py`
  - 作用：Baseline 预处理（PDF -> 普通切片，无摘要）
- `02_baseline_embed.py`
  - 作用：Baseline 向量化并构建对照索引（`baseline_data/indexes/`）
- `04_baseline_benchmark.py`
  - 作用：Baseline 同口径评测（DRM / JurAcc）

### 评测与回归

- `04_benchmark.py`
  - 作用：SAC 系统评测（支持 CSV 导出）
- `tools/api_smoke_regression.sh`
  - 作用：API 连通性与关键接口回归（4 项 smoke tests）

### 开发辅助 / 过程材料（非主流程）

- `dev_support/`
  - 包含：下载清单（manifests）、历史批量题集、辅助脚本、历史报告等

## 三、流程图版目录说明（直观版）

```text
crypto-reg-rag/
├─ raw/                              # 原始官方 PDF（按法域分目录）
│  ├─ hk/ sg/ uk/ uae/ ...          
│  └─ ...
│
├─ 主系统（SAC 路线）
│  ├─ 01_sac_pipeline.py             # raw/ -> SAC切片（摘要增强）
│  ├─ 02_embed_build_openai.py       # SAC切片 -> indexes/
│  ├─ indexes/
│  │  ├─ faiss.index                 # 主索引（SAC）
│  │  └─ meta.jsonl                  # 主索引元数据
│  ├─ 03_retrieve_and_qa.py          # 检索/重排/问答核心
│  └─ 06_api_server.py               # HTTP API（/jurisdictions, /query）
│
├─ 对照系统（Baseline 路线）
│  ├─ 01_baseline_pipeline.py        # raw/ -> 普通切片（无摘要）
│  ├─ 02_baseline_embed.py           # 普通切片 -> baseline_data/indexes/
│  ├─ 04_baseline_benchmark.py       # Baseline 评测（DRM/JurAcc）
│  └─ baseline_data/
│     └─ indexes/
│        ├─ faiss.index              # Baseline 对照索引
│        └─ meta.jsonl
│
├─ 评测与结果
│  ├─ 04_benchmark.py                # SAC 系统评测（DRM/JurAcc）
│  ├─ tests/
│  │  └─ manual/
│  │     └─ test_set_core_hk_sg_uk_uae.json   # 当前核心金标集（主线）
│  └─ reports/
│     ├─ core_hk_sg_uk_uae_summary_v2.csv      # SAC+Auth 汇总
│     ├─ core_hk_sg_uk_uae_baseline_summary.csv# Baseline 汇总
│     └─ ... (逐题结果 cases.csv)
│
├─ 运行辅助
│  ├─ tools/
│  │  └─ api_smoke_regression.sh     # API 一键回归（4项）
│  ├─ .env.example                   # 环境变量模板
│  ├─ README.md                      # 项目说明（开发视角）
│  └─ README_一键运行说明.md           # 提交用运行说明（当前文件）
│
└─ dev_support/                      # 开发辅助 / 过程材料（非主流程）
   ├─ manifests/                     # deep research 下载清单与探测结果
   ├─ scripts/                       # 下载脚本、增量刷新脚本、老题集修复脚本等
   ├─ tests/                         # 历史批量题集与旧手工题集
   ├─ reports/                       # 历史 benchmark 输出（如 B01）
   └─ docs/                          # 结构说明笔记
```

### 一句话理解整个流程

- `raw/` 是原始法规 PDF 入口
- SAC 与 Baseline 分别构建两套索引（用于 A/B 对照）
- `03 + 06` 提供可调用 API 能力
- `04_benchmark.py / 04_baseline_benchmark.py` 输出对照结果到 `reports/`
- `dev_support/` 只放开发过程材料，不干扰主流程

## 四、运行环境

- 操作系统：macOS（作者环境）
- Python 环境（已验证可运行）：
  - `/Users/chenzheyang/anaconda3/envs/crypto_reg/bin/python`

说明：
- 系统 `python3` 可能缺少 `faiss` 等依赖，建议使用上述解释器。

## 五、环境变量配置

项目使用 GPTSAPI/OpenAI 兼容接口进行 Embedding 与问答，需配置 API Key。

```bash
cd crypto-reg-rag
cp .env.example .env
```

编辑 `.env`，填写：

```bash
GPTSAPI_API_KEY=你的_API_Key
```

加载环境变量：

```bash
set -a; source .env; set +a
```

## 六、启动 API 服务（终端 A）

```bash
cd crypto-reg-rag
/Users/chenzheyang/anaconda3/envs/crypto_reg/bin/python 06_api_server.py
```

启动后默认地址：

- `http://127.0.0.1:8000`

可用接口：

- `GET /health`
- `GET /jurisdictions`
- `POST /query`

## 七、API 一键回归检查（终端 B）

该脚本会自动执行 4 项检查：
- 获取法域列表
- HK 单法域查询
- HK+SG 混合法域查询
- 非法法域 `XX`（期望返回 400）

```bash
cd crypto-reg-rag
bash tools/api_smoke_regression.sh
```

## 八、SAC 主系统评测（核心金标集）

使用核心文档级金标集（HK/SG/UK/UAE）进行评测，输出逐题和汇总 CSV。

```bash
cd crypto-reg-rag
/Users/chenzheyang/anaconda3/envs/crypto_reg/bin/python 04_benchmark.py \
  --test-set tests/manual/test_set_core_hk_sg_uk_uae.json \
  --out-csv reports/core_hk_sg_uk_uae_cases_v2.csv \
  --out-summary-csv reports/core_hk_sg_uk_uae_summary_v2.csv
```

结果文件：

- `reports/core_hk_sg_uk_uae_cases_v2.csv`
- `reports/core_hk_sg_uk_uae_summary_v2.csv`

## 九、Baseline（非SAC）对照评测（同口径）

与 SAC 使用相同核心金标集，输出可直接对照的 CSV。

```bash
cd crypto-reg-rag
/Users/chenzheyang/anaconda3/envs/crypto_reg/bin/python 04_baseline_benchmark.py \
  --test-set tests/manual/test_set_core_hk_sg_uk_uae.json \
  --out-csv reports/core_hk_sg_uk_uae_baseline_cases.csv \
  --out-summary-csv reports/core_hk_sg_uk_uae_baseline_summary.csv
```

结果文件：

- `reports/core_hk_sg_uk_uae_baseline_cases.csv`
- `reports/core_hk_sg_uk_uae_baseline_summary.csv`

## 十、A/B 对照结果查看（汇报用）

对照这两个汇总文件：

- SAC+Auth（v2）：`reports/core_hk_sg_uk_uae_summary_v2.csv`
- Baseline（非SAC）：`reports/core_hk_sg_uk_uae_baseline_summary.csv`

关键指标：

- `DRM`（越低越好）：Top-K 中错误文档比例
- `JurAcc`（越高越好）：Top-K 中目标法域比例

## 十一、如果需要重建索引（可选）

### SAC 主索引（较耗时）

```bash
cd crypto-reg-rag
/Users/chenzheyang/anaconda3/envs/crypto_reg/bin/python 01_sac_pipeline.py
/Users/chenzheyang/anaconda3/envs/crypto_reg/bin/python 02_embed_build_openai.py
```

### Baseline 索引（对照实验）

```bash
cd crypto-reg-rag
/Users/chenzheyang/anaconda3/envs/crypto_reg/bin/python 01_baseline_pipeline.py
/Users/chenzheyang/anaconda3/envs/crypto_reg/bin/python 02_baseline_embed.py
```

说明：
- 若仅需跑核心对照（HK/SG/UK/UAE），可先限定法域（示例）：

```bash
BASELINE_JURISDICTIONS='hk,sg,uk,uae' /Users/chenzheyang/anaconda3/envs/crypto_reg/bin/python 01_baseline_pipeline.py
```

## 十二、提交说明（邮件）

- 收件人：`dingxiaowei@nju.edu.cn`
- 邮件标题：请标注 `最终版`

建议邮件正文简述：
- 附件包含项目报告 + 可一键运行代码
- 已附一键运行说明
- API Key 出于安全原因未包含在附件中，请自行配置 `.env`

## 十三、安全提醒

- 请勿在提交包中包含真实 API Key
- 建议仅保留 `.env.example`，不要提交 `.env`
