# JEV Workbench / JEV 实验工作台

一个本地运行的文本分类实验工作台，把 JEV、Gemini、TF-IDF + XGBoost、TF-IDF + LightGBM 和固定规则放在同一个页面里。可以选择场景、开始或停止实验、查看逐条输入与预测、导出 JSON；运行结束的记录会保存到磁盘，刷新页面或重启后可读取。

仓库包含工作台代码、固定合成样本模板和 2026-10-07 修正实验的 15 组历史结果。启动服务只读取历史记录，不会自动调用远程 API。

## 本地启动

需要 Python 3.10+（本次验证为 Python 3.12）。后端与静态界面没有构建步骤。

```bash
git clone https://github.com/Xinyuan-Gao/jev-workbench.git
cd jev-workbench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jev_workbench.backend.server
```

- 原工作台：<http://127.0.0.1:8765/>
- 修正实验结果与样本复核：<http://127.0.0.1:8765/corrected.html>

`WORKBENCH_HOST` 和 `WORKBENCH_PORT` 可修改监听地址与端口，默认只监听本机。固定规则模式只需要 Python 标准库；依赖安装失败时，仍可启动服务和浏览历史结果，但树模型不能运行。macOS 上 XGBoost / LightGBM 若报告缺少 `libomp`，可先通过 Homebrew 安装：`brew install libomp`。

## 远程模型配置

```bash
cp jev_workbench/.env.example jev_workbench/.env.local
```

编辑 `.env.local`，填自己的 API URL、key 与 Gemini 模型 alias，然后重启服务。环境变量优先于这个文件；密钥由后端读取，不经过前端。仓库没有附带可用密钥。选择 JEV / Gemini 并点击开始会调用你配置的服务，产生的费用由提供商计费。规则、XGBoost 和 LightGBM 在本机运行。

示例配置沿用本次实验的中转地址。JEV 网关 `/chat/completions` 使用唯一 user message 内的 `{state, questions}` envelope，官方 TypeSafe 路由可配置 `/systemone`；只有对接真正的 OpenAI JSON 协议代理时才设 `JEV_API_PROTOCOL=openai`。Gemini 走 OpenAI-compatible messages 协议，使用独立 key。历史 alias `jev-latest` 和 `gemini-3.8-flash` 并没有被验证为不可变的底层模型版本，也不能仅凭 alias 确认官方产品身份。

## 样本数量与实验口径

| 场景 | 完整目录 | train | validation | test |
|---|---:|---:|---:|---:|
| 中文意图分类 | 216 | 126 | 36 | 54 |
| RAG 片段相关性 | 200 | 120 | 40 | 40 |
| Agent 路由 | 216 | 126 | 36 | 54 |

样本由固定模板与变体确定性展开，约 200 条指完整目录，不是每个场景正式测了 200 条。按 group ID 排序切分，组内变体不跨 train / validation / test；split 函数里的 seed 参数没有参与随机化。validation 留出，但本轮没有用来调参。传统模型及 TF-IDF 只在 train 上拟合，所有模式都只在相同 test 上评分。Gemini 请求不包含标准答案；JEV 的 RAG 请求同时包含 query 与 passage。

五种模式 × 三个场景，共 15 组历史运行都已执行结束。test 是 148 条记录、22 个核心模板/主题组；五种模式合计 740 次尝试，不是 740 个独立案例。Gemini 路由 54 条中，50 条返回合法类别、4 条返回过滤拒答，拒答单独计为失败。有效输出准确率和 Macro-F1 排除失败；全尝试正确率把失败计为未答对。

| 模式 | 意图：全尝试正确率 | RAG：全尝试正确率 | 路由：全尝试正确率 |
|---|---:|---:|---:|
| 固定规则 | 88.89% | 100% | 100% |
| XGBoost | 50% | 50% | 37.04% |
| LightGBM | 37.04% | 50% | 25.93% |
| JEV | 98.15% | 100% | 100% |
| Gemini | 100% | 95% | 92.59% |

完整统计见 [matrix.md](jev_workbench/reports/corrected_20261007/matrix.md)，逐条结果与分组清单在同目录。数据是合成模板、单轮运行；规则设计参考过整个目录的词汇。这些分数不能直接作为真实业务的泛化能力排名。历史结果里的四条拒答通过离线输出验证记为失败，保留原返回；当前逐条 runner 遇到异常会结束该组，不会自动补测未处理样本。重新运行时应检查状态和已处理数量。

## 项目结构

```text
jev_workbench/
  backend/       HTTP API、适配器、固定样本、树模型、评分与运行保存
  frontend/      无构建依赖的工作台、修正结果查看页
  tests/         请求答案泄漏、RAG 完整输入、分组切分和结果恢复检查
  reports/corrected_20261007/  已核对的历史结果与切分清单
  .env.example   空密钥配置模板
```

历史 JSON 已随仓库提供，新生成的运行记录默认被 Git 忽略。原始旧实验日志、本地凭据、飞书文档导出和私人工作区内容没有上传。

## 本地检查

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q jev_workbench/tests
node --check jev_workbench/frontend/app.js
node jev_workbench/frontend/smoke_check.mjs
```

Node.js 只用于前端静态检查，不是启动服务的必需依赖。上述 tests 使用本地替身，不调用远程 API。
