# Scientific runtime 只读审查

2026-10-07。审查 journal.py、remote.py、tests/test_runtime.py 与 PROTOCOL.md。未改生产文件，未调用真实 API。离线原有测试：32 passed in 3.62s。预算、真人路由 gold、实网 JEV schema/version canary 尚未完成，不把这些测试称为实验完成。

## 必须修复的实际问题

### P1 journal 异常字符串绕过脱敏

位置 scientific/journal.py:165-167。

最小触发：predict_fn 抛 RuntimeError('Authorization: Bearer FAKE_SECRET_KEY')。execute_jobs 将 error 原样存入 outcome；journal.records() 可读出完整假 key。这绕过 remote 的 _safe，因为 predictor 在返回安全result前就可抛异常，其他本地模型/配置异常也可携带凭据。现有网络异常脱敏测试并未涵盖该路径。

建议：统一的 credential-aware redaction 在 journal 持久化边界执行，包括 exception string、outcome、attempt；至少错误信息默认只记录 error_type + 无敏感内容的代码，另受控脱敏错误详情。若使用 redactor callback，必须强制部署runner注入实际credentials，不能只遮Bearer而保留纯 key 子串。测试包括 Bearer/header/plain key、nested outcome/attempt secret，并验证 SQLite原文也无secret。

### P1 未标记或处理恢复后重复请求

位置 scientific/journal.py:129-135 与 scientific/remote.py:261-265、298-309。

最小触发：一个 planned job 的predict_fn第一次中断（KeyboardInterrupt），恢复后成功；记录只有两个普通 job.started，不含 recovery、previous_attempt_id、in_doubt 或 potential_duplicate。

更实际危险路径：HTTP 已成功/已计费，finished callback 已落盘，process在record_terminal前终止。outcome仍NULL，resume会再次付费调用。若第1次合法结果已经有durable raw body，第2次结果覆盖为logical outcome，会引入未预注册的重测；若第一次响应没落盘，则重复付费量未知。

建议：每job执行有execution_id；HTTP attempts明确关联execution_id。恢复检查最后unfinished started或finished但未terminal，写recovered_from/potential_duplicate/previous_attempt_ids。对已经完整finished响应可用冻结parser恢复terminal而不再次调用；对in-flight状态记录indeterminate，由预注册策略决定是否重提，若重提所有实际attempt都计成本并显式说明。不要声称exactly-once；网关无idempotency时做不到。测试crash-after-HTTP-finish-before-terminal、crash-before-response-finish、记录恢复标记、不中断已terminal job。

### P2 已观察过的 Gemini 过滤拒答被误分为 malformed_response

位置 scientific/remote.py:23、175-180。

最小触发：parse_response('gemini', {'choices':[{'message':{'content':"This request was blocked by Gemini's filters. They can occasionally trigger by mistake on safe coding, security, or biology-related queries. Please try rephrasing your prompt."}}]}, ['yes'])。

实际status为malformed_response。原实验audit.json中的4条拒答都是这个真实形式。_REFUSAL只覆盖content/safety开头的blocked/filter，未覆盖request was blocked by ... filters。

建议：增加很具体的该已知网关过滤模式，避免单纯把所有"blocked"都当拒答。直接把audit保存的完整文本作为回归fixture；继续保留合法prediction的reason引述拒答不是实际拒答的测试。主accuracy分母目前仍计失败，所以总正确率不变，但refusal taxonomy和“有多少过滤”会错。

### P2 合法预测因置信度 overflow被抹掉

位置 scientific/journal.py:155-161。

最小触发：predict_fn返回 {'status':'valid','prediction':'yes','confidence':10**400}。float(confidence)抛OverflowError，内层仅catchValueError/TypeError，外层改整个记录为execution_error，prediction=None。

remote.parse_response已经catchOverflowError，但journal自己的归一化并没有。理论上可由合法本地prediction或自定义predictor触发，也违背协议“非法confidence不抹掉类别”。

建议内层增加OverflowError并保留prediction='yes'、status='valid'、confidence=None、confidence_status='invalid'，加入journal回归测试。不得把异常confidence推成 missing。

## 已通过的具体检查

- manifest全量planned表和records left-join outcomes，逐条ordinary Exception继续，不再用只成功rows缩分母。
- terminal overwrite幂等，不重跑已terminal；protocol/manifest hash不一致拒绝恢复。
- 可对http每attempt started/finished通过callback即时持久化，真实请求JSON、HTTP status、raw body、usage、response model保留；网络/429重试最多2次。
- remote对所注入实际api key、JSON secret字段、Bearer/Authorization/API-key文本响应和嵌套JSON字符串做脱敏，响应headers只白名单。
- JEV解析answers.classification.choice、named type=choice嵌套结果、OpenAI content JSON；合法类别+缺失/NaN/bool/out-of-range confidence在remote保持valid。
- malformed/refusal/invalid_label不改写不自动retry，费用明确unknown。

## 不阻断离线建设但须补充的验收边界

1. 现有“JEV真实格式”依据旧adapter/解析形状，没有原始HTTP历史fixture（旧实验没保存raw）。这些单测只证明可解析构造样本，尚不证明definitions字符串criteria被当前网关支持，不能绕过dev canary。实网canary必须用于dev非test，保存实际request/response后再冻结。
2. 成功HTTP返回finish_reason没有独立metadata字段，但完整raw body仍保存，可复算；终止原因为length的合法类别目前也整体判truncated_output。是否以strict output contract为失败须明确，避免对接收有效类别的业务口径模糊。
3. endpoint/模型alias目前在返回metadata而不在started event；若在首次请求中断，durable started不含endpoint、provider、requestedmodel，难以审计“谁被实际提交”。建议HTTP started事件含脱敏endpoint/provider/requested model/timeout、execution_id；timeout类型单独结构化，以便报告timeout与其他network_error。
4. Protocol status枚举未被journal严校验，record_terminal只拒绝pending，unknown/running可使complete=True。实际HTTP parser返回已知枚举，但部署入口的本地predictor/错误桥接应有允许status集合验收，不允许running作为terminal。
5. 单writer是明确前提，journal无跨进程领取/lease；不可用同DB并行启动两个execute_jobs后声称安全幂等。上线CLI可加单writer锁，或明确禁止。当前测试无需证明未设计的多writer。

结论：运行层基础能力已成立，但上述四项修复和恢复测试必须在正式调用前完成；科学数据、真实人工标注、版本/预算门槛仍独立pending。

## 修复后二次复核

已确认原四项修复：journal持久化边界 _redact 支持实际secret注入，尝试/异常/outcome均脱敏，status白名单拒绝running；OverflowError只使confidence invalid并保留合法prediction；真实Gemini blocked-by-filters wording返回refusal；HTTP started包含endpoint/provider/request_model/timeout/execution_id，finished包含冻结parser outcome。重新构造journal时，已完整finished可直接恢复terminal，未finished会落indeterminate/potential_duplicate且默认不再调用，需要显式allow_indeterminate_retry。上述设计符合逐次attempt和全planned分母要求。

额外发现尚需修复的恢复边界：_recover只有constructor调用。同一journal对象发生KeyboardInterrupt后不重新构造、再次execute_jobs，则未finishedHTTP仍被直接提交，无indeterminate marker，默认allow_indeterminate_retry=False也无效。最小触发是创建j、append一个phase=started HTTP event，直接execute_jobs(j,...): predict_fn被调用且complete=True。CLI每次重新进程构造journal路径安全，但公开runtime或未来工作台复用journal不能依赖这个偶然前提。建议execute_jobs入口事务refresh recovery后再pending_jobs，新增同对象unfinished与finished两种恢复tests。

指定statistics/datasets/runtime联合suite在二次复核时跑出61 passed (5.82s)，源码同时被root团队持续编辑，因此不声称这次结果覆盖测试运行开始后新增的case。fullscientific suite另看到local_models三个新持久化/labelsupporttests失败（3failed/68passed）；不属于本审查修改范围，已通知root团队，不能声称全项目通过。

部署必须把真实credentials注入DurableJournal(secrets=...)，否则独立纯key异常无法仅靠header模式完全遮蔽。该注入不是调用API授权，仅为日志防泄漏。singlewriter前提仍明确，不称跨进程exactly-once。MASSIVE候选结构和词面筛查通过也不替代正式API canary、真实gold和预算冻结。
