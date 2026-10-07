# 科学对比实验 v2

这套模块用于重新验证 JEV/Gemini 网关与本地监督分类管线，旧实验不参与本轮评分。

正式阶段 A 包括有公开人工标注的 MASSIVE 中文意图、T2 问题—片段相关性。路由仍属阶段 B，必须完成真实独立人工标注、裁决与独立拆分后再建立新冻结记录。两个阶段不共享事后挑出的假设，也不冒充三场景均已完成。

## 运行环境

从 `/Users/gxy/投稿管理` 执行以下命令。已训练权重来自 Python 3.12.0，精确包版本见 `requirements.lock.txt` 和 `audits/runtime_environment.json`。Joblib 权重仅加载本项目自己的训练产物。

```bash
python3 -m pytest -q jev_workbench/scientific/tests
python3 -m jev_workbench.scientific.design_sensitivity
python3 -m jev_workbench.scientific.run --status
python3 -m jev_workbench.scientific.status --watch
```

进度页：http://127.0.0.1:8765/scientific.html 。旧服务继续提供该页面；状态发布器只发布白名单汇总，不公开标准答案、任务原文、原始 HTTP 或密钥。

## 数据与重建

原始来源、许可和采样审计保存在 `data_sources/`。MASSIVE 数据 CC BY 4.0，代码 Apache 2.0；T2 数据 Apache 2.0。原始源及候选清单不因名称包含 test 就自动获得科学可信度，需要核查原始标签与拆分。公开模型预训练污染不可排除。

本轮候选：MASSIVE 训练 10,684 / 验证 1,958 / 测试 558 条；T2 训练 1,200 / 验证 200 / 测试 200 个问题，对应 4,030 / 671 / 674 个片段对。MASSIVE 测试按类限额，不保留自然频率；T2 是原公开 dev 内的 query 拆分，不是官方 test。

以下代码重建记录并核对候选内容哈希，不训练或预测：

```python
import json
from pathlib import Path
from jev_workbench.scientific.datasets import (
    load_massive, load_t2_source, clean_near_duplicates, manifest_hash,
)
root = Path('jev_workbench/scientific')
massive = load_massive(root/'data_sources/massive/zh-CN.jsonl',
                       test_cap_per_label=10, seed=20261007)
massive['records'], removals, near = clean_near_duplicates(massive['records'])
t2 = load_t2_source(root/'data_sources/t2/selected_original_records.jsonl',
                    seed=20261007)
for name, candidate in [('massive', massive), ('t2', t2)]:
    saved = json.loads((root/f'data/{name}_candidate.json').read_text())
    assert manifest_hash(candidate['records']) == manifest_hash(saved['records'])
```

已经实际执行此重建，核对结果见 `audits/dataset_reproduction.json`。重新下载完整 T2 collection 后，须核对官方 SHA256 再提取 pid；当前持久副本保存全部选中源记录，可复算本轮清单，不能声称包含整个 3.66GB 检索库。

## 训练与选择

`local_models.train_local_models(train_records, validation_records, labels, output_path=...)` 仅接收训练与验证记录。多数类固定；LR、LinearSVC、XGBoost、LightGBM 各 12 个预定配置。特征拟合只使用 train，dev Macro-F1 决定配置，平分按预设候选序号。随机树模型保留 5 个真实种子；不能复制相同预测增加重复次数。

两份 `artifacts/<task>/local_dev_report.json` 保存全部候选、选择理由、输入 ID、版本与耗时；对应 `local_models.joblib` 保存真实权重。T2 曾有稀疏特征变量错误，修复后原网格完整重训，中断报告单独保留。本轮最佳本地确认性对象已在 test 前选定：MASSIVE 为 LR，T2 为 LinearSVC。

## 冻结与正式调用

`configuration_candidate.json` 是候选；只有 `frozen.json` 才是不可覆盖的本地冻结记录。冻结前必须核查全部 gates，并保存协议、数据、schema、dev 报告、权重、源码、依赖与审查证据。`metadata.execution_source_sha256` 强制绑定实际运行源码，不能用另一个快照代替当前执行版本。冻结 hash 证明内容一致，不证明外部时间认证、人工标签完美或模型官方身份。

阶段 A 总矩阵 23,408 个组合：本地 16,016 个，远程 7,392 个。每个远程组合最多 3 次网络尝试，总上限 22,176 次，默认禁止不确定请求静默重发。次数限制能阻止额外请求，无法在单价未知时保证人民币金额上限。网关实际费用标为未知。

冻结之后才可以执行：

```bash
python3 -m jev_workbench.scientific.run --local
python3 -m jev_workbench.scientific.run --remote --env-file jev_workbench/.env.local
```

每任务保存完整 `matrix.sqlite` 与汇总。恢复执行同一命令只处理 pending，不重新拟合、不改参数。退出码：0 为选择的执行过程结束（还须查看全矩阵 pending）；2 为配置或冻结错误；3 为调用上限停止；4 为审计持久化失败；130 为中断。退出 0 不代表所有输出都是合法类别。

HTTP started 在发送前持久保存，finished 保存原始响应与 usage。记录写入失败会停止并保留 pending，不能变成普通错误样本后称完成。若已保存最终响应，恢复直接生成终态；若是否付费完成不确定，会保留 indeterminate，不能自动再付费。网络和临时 HTTP 错误可重试；拒答、错误标签和解析失败保留，不改写输入以挑成功。

## 统计与解释

主要比较预设 repeat 0；重复用于稳定性，不增加独立 n。全提交 Accuracy/Macro-F1 把拒答与失败记为错误/对应类别漏报，valid-only 分数单列其分母。意图固定 60 类 Macro-F1，并另报有支持 59 类口径；稀有类不能由总数掩盖。RAG 主正确率按片段对，另报 query 等权平均；重采样与检验均保留 query 组。

正式输出由 `analyze.py` 从只读 SQLite 重算，不信任网页上的旧分数。矩阵、gold、repeat 或终态不完整会拒绝最终分析。每任务确认性比较为 JEV–Gemini 与 JEV–dev 最佳本地管线；全任务组成一个 Accuracy Holm 家族，其他指标标为探索性。置信区间固定 10,000 次 bootstrap，RAG 配对置换固定 10,000 次。

`audits/design_sensitivity.json` 是预先假设的样本量分析，不用正式预测估计 discordance。558 条不能保证识别 5pp 差异；200 个 query 不能凭数量声称具有 80% 检验功效。未显著不等于等效，不事后扩样追求显著结果。

最终文章需要完整终态、独立重算、真实截图和错误依据后才改结论。API response model 只是网关自报字段，本轮比较对象是所记录的服务和配置。本地监督管线与远程零样本服务的训练资料、算力与前置成本不相同。

## 路由标注与发布边界

`data_sources/routing/route_annotation_A.csv` 和 B 为同一批 1,600 个去重真人首任务、不同随机顺序、没有模型建议；标注人分别使用 `GUIDE.md`。先独立提交，再计算一致率/kappa，由第三方或共同复核裁决。不能把两次 subagent 标注写成人工双盲。真实标注完成之前没有路由标准答案。

公开仓库不得包含 `.env.local`、私人飞书导出、未经逐项隐私审查的 WildChat 原文、API 原始请求或权重大文件。公开来源许可不代替个人信息审查。内部证据路径与公开发布包分开管理。
