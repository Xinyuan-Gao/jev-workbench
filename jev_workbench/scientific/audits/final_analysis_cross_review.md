# 最终分析入口交叉审查（修复后复核通过）

2026-10-07。只读审查 `analyze.py`、`tests/test_analyze.py` 与 `configuration_candidate.json`，未修改分析代理源码，未预测正式 test、未调用真实 API。复现只使用系统临时目录下的合成 journal；真实候选仅读取配置与 dev 报告。

当前结论：2026-10-07 修复后只读复核通过。四个原始分析边界均用新建合成 fixture 独立重放，前三项拒绝且不写 final，第四项返回结果与输出文件都脱敏；分析及 CLI 测试共 61 passed in 2.28s。下文保留首次发现与要求的历史，逐项记录最终实证。此次覆盖的合同未发现阻塞；这不是正式 test 结果或全仓库测试声明。

## 首次发现与修复后实证

### P1：缺 HTTP 证据仍允许 final

**当前状态：已修复，独立复核通过。** 删除所有远程 attempts 后拒绝，错误为 `Complete remote HTTP audit evidence is missing/unpaired`；没有创建输出文件，SQLite 字节保持不变。最新实现核验配对请求/响应、执行身份、冻结 payload、重试上限及 raw response 重新解析。

`_read_matrix` 只要求 terminal 和冻结 manifest 相符；`_operational` 对缺 attempt 只汇总计数。在完整合成矩阵中删除全部远程 attempts，`analyze_experiment(...)[complete]` 仍为 True。只删除一条远程 finished 的已新增回归也失败：未抛 ValueError。

应核对远程记录的 started/最终 finished、对应 attempt/execution ID、请求体、provider/requested model、HTTP status、raw body 等证据。网络失败允许状态为 None 和空 raw body，但字段与事件必须明确。已授权 indeterminate 恢复中的未知旧 attempt 要单独报告，不能冒充其响应/usage 已完整取得。

### P1：terminal 与 HTTP outcome 矛盾仍允许 final

**当前状态：已修复，独立复核通过。** 将远程 refusal terminal 改为 valid/yes、保留 HTTP outcome 后拒绝，错误为 `Logical terminal outcome differs from final HTTP outcome`；没有创建输出文件，SQLite 字节保持不变。raw body 与解析 outcome 的一致性另有通过的回归覆盖。

合成 fixture 中把一个远程 refusal terminal 改成 valid/yes，保留 finished.outcome=refusal，仍接受 complete=True。这会让分类成绩与实际服务证据不一致。

应要求 terminal 的 status/prediction/confidence 语义与最后完成的非 retry HTTP outcome 一致，并核对请求体是冻结输入、任务定义与 frozen model 构建的白名单 payload。允许仅追加 recovered/transport 等审计字段。

### P1：没有强制分析入口自身源码 hash

**当前状态：已修复，独立复核通过。** 移除 analyze.py 的 source mapping 和 artifact 并重新冻结合成 bundle 后拒绝，错误为 `Actual analyze.py source must be included in the frozen source mapping`；没有创建输出文件，SQLite 字节保持不变。最新实现直接比较本次模块 `__file__` hash；替换复制件而不匹配实际执行模块的回归也通过。

原 `test_analyze` fixture 的 execution_source_sha256 只含 run 所需九份代码，不含 analyze.py，仍可产生 final。run 对额外 analyze.py 映射会检查，但“可选额外”不能代替分析入口强制校验。

分析入口应要求 analyze.py 出现在 source mapping 与 frozen artifacts，并直接核验本次分析模块实际 `__file__` 的 SHA256；不能仅核验从未执行的复制件。

### P2：最终 JSON 的同值凭证副本泄漏

**当前状态：已修复，独立复核通过。** 在本地 failed outcome 注入合成 `api_key` 与同值 `echo`（避开远程证据一致性拒绝，以单独测试输出边界），分析正常完成，但返回结果与写出的 JSON 均不含合成敏感值；SQLite 字节保持不变。最新最终输出调用 `_safe(result, _secrets(result))`，未读取真实凭证。

直接在合成 refusal outcome.raw_output 中写 `{api_key: SYNTHETIC_REVIEW_SECRET, echo: 同值}`，最终 `_safe(result)` 遮蔽 api_key 字段，但 echo 仍保留同值。正常 runner/journal 的注入脱敏可提前防住该路径，本例证明分析输出边界自身尚不完整，未涉及真实凭证。

建议分析最终输出统一使用从结果递归提取的敏感值进行脱敏（例如 `_safe(result, _secrets(result))`），或拒绝未经脱敏的历史记录；不应为此读取真实 credential 文件。

### P1：正式网关地址尚未冻结（已转运行代理修复）

冻结执行身份原来只指定 remote_models，credentialmap 可注入任意 base_url，等于允许正式运行改换服务宿主。应冻结 `execution.remote_base_urls`，限定合法 HTTP(S) 地址、拒绝 userinfo/query/fragment，注入地址标准化后必须相符；API key 不进入配置与冻结。

主代理随后要求暂停分析复核、先修运行边界。该项现已离线修复：`run._frozen_inputs` 强制两家 remote_base_urls，拒绝非 HTTP(S)、userinfo/query/fragment、非法 host/port 与凭证型路径；远程注入 URL 标准化后必须和冻结值一致，client 使用冻结地址。新增 12 个失败案例先 red 后 green，尾部 slash 相符与 localhost HTTP mock 正常接受。此前 run/runtime/local 合计 97 passed in 8.33s，run/test_run 编译通过；本轮 37 项 CLI 回归再次通过，没有访问真实网关。当前 candidate 已包含 remote_base_urls；实际冻结 gate 由主代理验收，API key 不进入候选配置。

## 已核实的正向合同

- `_read_matrix` 使用 read-only SQLite snapshot，比对完整冻结 planned job JSON、manifest hash 与 protocol hash；缺组合、缺任意 repeat、改 gold/input/model 都不能通过 manifest 等式。
- 全部任务先验收 terminal，再计算或写最终结果，不以成功子集替代 planned 分母。原始拒答 fixture 的全提交 accuracy 为 0.5，valid-only accuracy 为 1，二者有不同分母。
- 主比较只取 repeat=0；本地 seed 与远程实际重复进入描述性 stability 和 operational，不扩大独立样本数。
- 每任务确认性比较固定 JEV–Gemini 与 JEV–验证集最佳本地；所有正式任务共用 Holm family。两任务合成验证 family size=4，而非每任务各自校正。
- 独立一行组用 exact McNemar；同 query 多 passage 使用 group sign permutation 和按组 bootstrap。mean_group_accuracy 独立报告点估计，不挪用行加权区间。
- 固定完整 labels 的 Macro-F1 与 supported-label Macro-F1 分开；无 test 支持的类别仍在固定分母中记 0。
- 当前真实 candidate.analysis 可被 `_settings` 解析：primary_repeat=0，bootstrap/permutation 各 10,000，seed=20261007，alpha=.05；MASSIVE 按 gold_label 分层、best local=lr；T2 按 query group、无 label 分层、best local=linear_svc。与两份当前 dev 报告的 validation 选择一致。
- 费用为 unknown，missing usage 不填 0；结果附公开污染、罕见类别退化区间、重复不增加 n、无通用排名等边界。

## 验证与范围

首次快照 `test_analyze.py` 为 10 passed、1 failed；该记录是修复前历史。最终运行 `python3 -m pytest tests/test_analyze.py tests/test_run.py -q --tb=short`，结果 **61 passed in 2.28s**（分析 24、CLI 37）。四项独立重放目录为 `/var/folders/r7/05tz3xrs6hgc05yq9xzk10nc0000gn/T/jev-final-cross-review-864q93wc`，仅含合成数据。

本轮复核源码 SHA256：

- `analyze.py`: `99e4bc299825c55fe16b293d3e998b30685c492169829f6cc672358c7e8d2057`
- `tests/test_analyze.py`: `4bfe8c1328be3844e67382aa5e599056a463a05188c6254a631f1f7cc9dfbd6b`
- `run.py`: `2ccf7fc6caa36e8a24f2373c6b27d37375c32fde5e5cb9cf9e3da6157558b3e4`
- `tests/test_run.py`: `1d853725706bc446e0e61c1cdced82261e21dc492fc6f46d1480c3e941514daa`

本轮只更新审查报告，未修改稳定源码，未访问真实 API、未预测正式 test、未读取真实 env，也未把合成分数作为真实实验成绩。完整全仓库回归与最终冻结由主代理另行完成。

范围限制：当前候选禁止 indeterminate 重发，因此要求每个历史 HTTP started 都有 finished 与本次计划一致；允许未知旧请求的恢复策略不在本轮支持范围。审查覆盖当前 canonical 网关配置，未将非 canonical 的同义 URL 拼写扩展为分析兼容性承诺。
