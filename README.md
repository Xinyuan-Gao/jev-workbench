# JEV Workbench / JEV 对比实验工作台

把 JEV、Gemini 和本地文本分类管线放到同一批测试上比较的实验代码、执行前协议、审计记录与结果。所有成绩都来自本仓库里的运行记录，可以按下面的命令从只读的 SQLite 重新算一遍。

## 这次实验做了什么

阶段 A 用两份有公开人工标注的中文数据：

| 任务 | 数据 | 独立测试单元 | 标签 |
|---|---|---:|---|
| 中文意图判断 | [MASSIVE 1.1](https://github.com/alexa/massive) zh-CN | 558 条话语 | 60 类，测试覆盖 59 类 |
| 问题—片段相关性 | [T2Ranking](https://github.com/THUIR/T2Ranking) | 200 个问题 / 674 对 | relevant、irrelevant |

比较对象是五种本地管线和两个远程服务（走同一家第三方兼容网关，零样本）：

- 多数类地板线
- TF-IDF + Logistic Regression
- TF-IDF + LinearSVC
- TF-IDF + XGBoost
- TF-IDF + LightGBM
- JEV（请求别名 `jev-1.13.0`）
- Gemini（请求别名 `gemini-3.8-flash`）

本地管线各自在训练集拟合、在验证集按 Macro-F1 选配置（每个可训练模型 12 个预定配置，树模型另跑 5 个真实种子）；远程服务每个组合跑 3 轮真实调用。整个矩阵 23,408 个样本×模型×重复组合，其中本地 16,016 个、远程 7,392 个，本轮实际发起 7,769 次 HTTP 请求。

## 主要结果

全提交正确率，primary repeat 0。失败计入分母，另有 95% 区间：

| 模型 | 中文意图（558 条） | 问题—片段相关性（674 对） |
|---|---:|---:|
| 多数类 | 1.79% | 55.79% |
| TF-IDF + LR | 79.39% | 61.28% |
| TF-IDF + LinearSVC | 79.21% | 61.57% |
| TF-IDF + XGBoost | 66.67% | 59.94% |
| TF-IDF + LightGBM | 64.34% | 59.50% |
| JEV | 78.85% | 60.53% |
| Gemini | **83.33%** | 61.72% |

四个预定比较（全提交正确率，Holm 校正）：

| 比较 | 差值 | 95% 区间 | Holm p |
|---|---:|---|---:|
| 中文意图 JEV − Gemini | −4.48pp | −6.81 ~ −2.33 | 0.0019 |
| 中文意图 JEV − TF-IDF + LR | −0.54pp | −3.58 ~ +2.51 | 1.000 |
| 相关性 JEV − Gemini | −1.19pp | −4.31 ~ +1.92 | 1.000 |
| 相关性 JEV − TF-IDF + LinearSVC | −1.04pp | −5.66 ~ +3.56 | 1.000 |

只有中文意图上 Gemini 显著高于 JEV。JEV 和本地线性基线在两项任务上都分不出方向。JEV 明显更好的地方是有效输出率（相关性任务 97.18% 对 Gemini 95.10%）和延迟（相关性任务均值 2.22 秒对 5.62 秒）。

## 仓库结构

```
jev_workbench/
  scientific/          实验模块：数据清理、模型训练与选择、冻结、运行、统计、图形
    PROTOCOL.md        执行前协议（研究问题、数据验收、指标与比较口径）
    README.md          模块说明与重建步骤
    frozen.json        冻结清单与源文件哈希
    frozen_continuation.json  续跑冻结（见「已知限制」）
    audits/            数据来源核查、验证集选择复核、样本量敏感性核算
    artifacts/         各任务的 summary、验证集调优报告、状态快照
    tests/             模块测试
  reports/jevsci_20261007/
    final_analysis.py  文章侧分析：复用冻结统计，容忍一条非首要轮次的 pending
    final_analysis.json
    make_article_charts.py + assets/   文章图表
    article_source.md  文章正文（带排版标记）
  backend/             本地工作台服务
  frontend/            静态页面，含 scientific.html 进度页
  tests/               工作台测试
article/               文章成稿
```

## 快速开始

需要 Python 3.10+（本次验证为 Python 3.12）。

```bash
git clone https://github.com/Xinyuan-Gao/jev-workbench.git
cd jev-workbench
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-dev.txt

python -m pytest -q jev_workbench/tests jev_workbench/scientific/tests
python -m jev_workbench.backend.server
```

- 工作台：<http://127.0.0.1:8765/>
- 本次实验状态页：<http://127.0.0.1:8765/scientific.html>

远程模型需要自己的凭据：`cp jev_workbench/.env.example jev_workbench/.env.local`，填入自己的 API URL、key 与模型 alias。密钥只由后端读取，不经过前端；仓库里没有任何可用密钥。本地管线不需要凭据。

## 数据来源与许可

原始数据不随仓库分发（体积和许可考虑），需要按下面的出处自行下载并用清单里的 SHA256 校验：

- MASSIVE 1.1，CC BY 4.0（数据）/ Apache 2.0（代码）：<https://github.com/alexa/massive>
  - 论文：<https://arxiv.org/abs/2204.08582>
  - 数据包：<https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz>
  - 使用文件：`data/zh-CN.jsonl`，SHA256 记录在 `scientific/data/massive_candidate.json`
- T2Ranking，Apache 2.0：<https://huggingface.co/datasets/THUIR/T2Ranking>
  - 论文（四级相关性标注）：<https://arxiv.org/abs/2304.03679>

重建与校验步骤写在 `jev_workbench/scientific/README.md`。仓库保留了数据清理后的任务 schema，但不含清理后的候选数据和训练权重，所以直接运行 `verify_freeze` 会报告缺失——这是有意的，不是哈希不一致。

## 实验口径与已知限制

- 测试前先冻结协议、清单、配置和源码哈希；测试标签只用于评分，模型输入按任务白名单构造，标签、分组 id 和记录 id 都不进请求。
- 只有全提交准确率做了预定比较和 Holm 校正；Macro-F1 区间标为探索性。未显著不等于等效。
- 一条 T2Ranking 的 Gemini 第 3 轮请求停在 `indeterminate`：请求已发出、结果不确定，冻结策略不允许为凑满矩阵重复提交可能已经计费的请求，因此保留 pending。它不在 primary repeat，不进入任何预定比较。`analyze.py` 会因为矩阵不完整拒绝出正式分析，`reports/jevsci_20261007/final_analysis.py` 是容忍这一条并显式记录排除范围的分析脚本。
- 远程延迟包含网络、网关排队和重试等待；本地延迟不含训练。两者不是同一个速度指标。
- token 用量有记录，但没有可核对的单价或账单，因此不给费用数字。
- 公开数据无法排除预训练污染；本地分词与去重只做词面筛查。
- 阶段 B（Agent 首步路由）需要两位真人独立标注与裁决，尚未建立标准答案，不在本次结果里。

## 文章

`article/JEV-public-benchmark-comparison.md` 是这次实验的文章成稿，包含完整的方法、结果表、错例和限制说明。
