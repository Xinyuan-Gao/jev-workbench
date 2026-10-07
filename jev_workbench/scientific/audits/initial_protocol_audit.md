# JEV 新一轮实验：独立方法审查与预注册验收建议

审查日期：2026-10-07。只读取本地 backend 与 corrected_20261007/audit.json；没有调用付费 API，也没有声称进行了新的人工标注或外部资料验证。

## 一、能回答的问题与研究范围

主问题应收窄为：在冻结的中文有限标签任务、指定训练资源和指定中转服务条件下，JEV 与 Gemini 以及经过合理 dev 调参的传统文本分类流水线，在正确分类率、合法输出率、延迟和可核实成本上有何差异？

研究对象是部署系统（输入定义 + adapter + 模型别名 + 网关 + 失败处理），不是无法确认的裸模型。用户关心可用性，网关拒答仍影响部署系统结果；但不能把网关拦截归因为底层模型必然拒答。

三场景分别报告，不以简单平均掩盖任务差异。代理路由是“按明确路由政策选下一项工具”的分类准确度，不等于执行该工具后的任务完成率；RAG 是片段判相关，不等于完整检索或问答系统效果。

预注册必须先于第一次正式 test 输出可见。预注册文件、数据 manifest、prompt、参数搜索空间、analysis 脚本分别记录 hash 与冻结时间。旧 test 已被反复查看，不再作为未见过的确认性 test；只能做开发或历史对照。

## 二、现有代码确认的问题

1. audit 记载 632 条合成记录、148 test，仅 22 个独立组：意图 9、RAG 4、路由 9。后缀变体高度相关，740 次尝试不等于 740 个独立观察。
2. experiments.split_records 按 group ID 排序切分，明确 del seed，不能描述成随机 seed 划分。
3. TraditionalModel 仅 char TF-IDF(2,5), max_features=3000，加固定 XGB/LGB 参数；dev 没用于调参，RAG 输入简单拼接，没有匹配/交互特征。
4. adapter JEV criteria 为英文标签映射到 None，instructions 泛化到“根据输入选择最合适类别”；Gemini 同样无细致、共享的语义标签定义。尤其多意图、多工具请求没有稳定政策时，gold 不是唯一合理答案。
5. adapter._request 不保留原 HTTP 请求/响应、usage、逐次重试与错误响应。仅 last_log 不能复算成本、确认版本或复查过滤原因。
6. runner._execute 单条异常会退出整个循环，finally 只保存成功 append 的 rows；metrics 的分母是这些 rows，而不是 planned test 全数。新一轮直接复用会产生严重丢分母问题。
7. runner.summary 把 stopped/failed 也视为 finished；“运行结束”和“全 planned 样本有可审计结果”要区分。
8. evaluation 的 macro_f1_completed 排除失败，不能作为全尝试主指标。avg/p95 latency 也只含 completed，会系统性隐藏失败耗时。
9. classify 把 confidence 无效也作为整条 ValueError。分类类别合法性与概率/置信度有效性应分开；否则未返回置信度的合法类别被额外惩罚。
10. 旧 audit 已识别 gateway alias 可变、confidence 不可直接比较、规则看过完整目录、旧延迟不含本地训练。这些边界仍须继承。

## 三、数据最低门槛

### 3.1 分开“确认性主实验”和“压力测试”

主实验优先有许可证、数据卡、清楚标签来源的真实公开数据和官方 train/dev/test；若官方 test 已被挑选查看，不得利用 test 输出设计 prompt。公开数据可能被预训练记忆，不因 heldout 自动消除污染。

压力测试可以包括新写的合成边界例、后缀/同义改写、拼写噪声、提示注入、多意图等；单独列出，不能混入自然分布主 test 后声称泛化到真实业务。新写合成例也必须记录生成方式、来源、审核身份。模型生成的 gold 不等于人工标注。

### 3.2 数量与独立单元

每个主场景至少 200 条冻结 test，但这是最低数量条件，不是充分严谨的证明。建议意图、路由分别 240 个不同原始请求，避免同一核心模板换尾巴；RAG 建议至少 200 个不同 query，每 query 同时含相关和困难不相关 passage（通常 >=400 pairs）。如果只取得 200 pairs / 100 query，只能写实际 n 和较宽 CI，不能称 200 个独立 query。

所有记录保留 source_id、document/conversation/query_id、group_id、label_source、license、split、duplicate_family、construction_method。组优先按共同原始文档、同一对话、同一 query 或共同生成模板建立，不能人为每条给独立 group 掩盖关联。

train/dev/test 保持整个组不跨区；有官方 split 优先沿用并审核重复。自建 split 用预先冻结的 stratified group random seed，记录分区原始名单；随机不保证代表性，仍报告来源、标签、长度、难度分布。跨 task 的同一来源也记录。

规范化 exact 去重（Unicode、空白、大小写）+ char ngram/词重合近重复检测 + 高相似候选复查；RAG 对 query 和 passage 均查，同一文章多个片段按文档分组。阈值在 dev 冻结，不为了漂亮 test 分数修改阈值。任何去重应记录删例原因，不能看模型错误以后只删不利例。

### 3.3 标签质量

官方 gold 可做主实验，清楚说明沿用何处的标签，并独立抽查歧义；不能把抽查称全量双人审核。对自建路由定义固定下一步政策，例如：需要实时外部事实 -> search；核心交付是可执行程序 -> code；纯内容组织/改写 -> writing。政策须明确混合请求、歧义请求、无匹配项怎样处理，不能事后按某模型输出改 gold。

可信自建 gold 理想要求两名真实标注者独立标注、分歧仲裁，保存匿名标注记录，并报告 agreement（例如 Cohen's kappa）、各类分歧比例。无人类参与时必须如实写自动/作者制定标签，不能让两个 subagent 充当“两位人类”或让 LLM 评审冒充客观 gold。

如果不存在可靠官方路由数据、也没有人类复核，自建路由应为探索性压力测试，确认性科学结论只覆盖有可靠 gold 的任务。可继续完成代码、素材和标注界面，但不能用“实验完成”跳过此门槛。

## 四、公平的系统配置和调参

### 4.1 传统文本模型

必须含真正强的文本基线：TF-IDF + LogisticRegression、TF-IDF + LinearSVC，再保留 XGBoost、LightGBM。树模型不是对稀疏文本天然更强，选择它们不能代替线性文本基线。

固定 dev 搜索预算并公开每个 candidate 的参数、dev score、耗时与选中原因。一个可行预算：每种模型每场景 12～24 个配置，选 dev macro-F1（failure-aware）的最高值，同分选更简/更快模型。不是要求模型参数完全相同，而是提供合理且透明机会。正式 test 不用于选特征、参数、阈值或 stop iteration。

TF-IDF 候选可包括中文 char ngram(1,2)/(2,4)/(2,5)、max_features 10000/30000、min_df 1/2、sublinear_tf；LogReg/SVC C 0.1/1/10，class_weight None/balanced；XGB depth 2/4/6、learning_rate 0.05/0.1、适当轮数；LGB leaves 7/15/31、min_child_samples 5/10/20、小数据不能不经审核固定默认20。完整笛卡尔积不必要，可预先定义平衡网格/固定随机搜索种子。

特征向量器、标准化、SVD（若使用）全部只在训练数据 fit；validation transform；不使用 test 文本 fit 词表。可冻结配置后 train+dev refit，但必须预先声明所有方法怎样利用 dev、记录最终 fit IDs；或为了最清楚选 train-only 最终 fit。两种都科学可行，不能混用不说明。

RAG 不允许只依赖无交互的 query+passage 拼接作为唯一传统基线。至少增加：共享 train-only TF-IDF 下 query/passage cosine、char/word/Jaccard overlap、长度比、关键实体/数字匹配；可用分离 TF-IDF 特征 + elementwise product/abs difference；可另设 BM25/cosine 固定基线，阈值仅 dev。公开消融“拼接 vs 交互”，避免归因到模型名称。

### 4.2 远程模型

相同任务信息：相同中文标签定义、问题、正文、候选类别、决策政策；都不带 gold、group、source 的答案暗示。native protocol 可以不同，JEV Choice 与 Gemini JSON不要求字节一致；需要验证语义信息相同，记录 native 协议与输出解析规则。

prompt 在 dev 选定，冻结所有请求示例/系统指令、候选顺序与最大输出 token、temperature/seed（只有服务支持时记录，不能假装被接受）；禁止看 test 错误后改 prompt 再把修改后成绩称预注册成绩。

如果要 few-shot，示例只能 train，且两远程共享示例和相同信息预算；zero-shot 主实验也可，但传统 supervised 与远程 zero-shot 是不同训练资源制度，应明说，不说“同样训练条件”。额外 few-shot 实验必须事先规划。

探测网关可用模型别名及返回 metadata，在结果里记录 alias、返回 model、provider revision/snapshot（如可确认）、网关与时间。若只看到可变 alias，诚实标记“无法独立确认底层版本”，研究对象限定该 alias 的服务系统。不得把配置名直接当官方发行版本。

### 4.3 规则

只在 train/dev 设计规则，冻结再 test；若看过 full test，降级为非盲参考。不能假定旧规则无监督且免费公平。规则也记录代码 hash 与开发资源。

## 五、运行和日志

隔离新实验目录/独立 runner，不停止现有 8765 工作台；旧记录保留且标注历史。新 runner 应有可恢复 durable journal，逐条调用前落盘 planned request ID，调用后写 terminal status；异常只影响该条，所有未执行条保留 pending/cancelled，禁止从分母消失。

每 logical item × model × repeat 保存：dataset/protocol/prompt/code/config hashes，input_id、group_id、gold（仅 evaluator可读）、repeat、seed、request order、start/end UTC、endpoint（无凭据）、exact non-secret request body、HTTP status、safe response headers、原始 response body、解析结果、finish_reason、usage、server request ID、error code、per-attempt duration、retry cause/backoff、total elapsed、billable units/verified price source。Authorization、key永不写日志。

自动检查 request body 的 gold/label/leakage field，人工/独立脚本比对所有信息字段；不是只查看 Python clean_item 名称。日志中 gold 可记录，但不能出现在发送给模型的 payload。

预冻结 timeout/retry：建议 first-attempt 作为“首次返回可靠性”主口径；运维可对408/429/5xx/transient network做最多2次重试，logical-item最终结果为补充。拒答、非法类别不通过改写重试。两种口径都报告，全部重试成本计入。重复跑不是选择最佳一次，不能只保留最后成功。HTTP错误日志可能含敏感信息，应脱敏后保存。

每主 test 远程建议3轮，按冻结的每轮seed/order交错随机执行 JEV/Gemini，同item配对，记录时间块。传统XGB/LGB建议3～5训练seed，线性确定性可只拟合一次。repeat评估系统不稳定性，不增加数据独立 n；不能把 240×3 说成720个独立case。主确认性结果预先指定round1，3轮均值/方差/一致率是稳定性补充，或预注册repeat平均的item级correctness为estimand并依group bootstrap，不事后挑更好定义。

## 六、指标、统计和判断规则

每条planned test必有terminal类别：valid_class / explicit_refusal / HTTP_error / timeout_network / malformed_response / invalid_label / truncated_output / cancelled_pending（最后一种说明实验尚未完整）。confidence_missing/invalid作为独立annotation，合法class仍可用于分类主分；若业务必须confidence，另外报告strict-contract rate。

主指标每场景 all-attempt accuracy = 正确合法类别 / 全planned logical items，失败计错；配 failure-aware macro-F1（合法gold各类的FN包括失败，失败不能错误变成额外gold类别），类别召回和confusion matrix含failure列。secondary valid-only accuracy/macro-F1与valid-output rate并列，不能只报valid-only高分。

报所有attempt和logical item延迟：median/p90/p95、总耗时、retry占比、成功/失败分层。本地报向量化+predict和fit/dev search耗时，硬件/线程数；远程报网关端到端，不能把1ms本地与1s远程称纯模型推理差异。没有真实usage/账单就写cost unavailable，不能默认0，不拿无usage冒充便宜。

置信度若没有同义、合法完整概率，不做跨模型置信度排名。ECE/Brier需符合概率约束且样本足够；校准只能train/dev，作为附加分析，不影响主分类分。

统计预注册：paired group bootstrap（例如10000次，固定分析seed）按真实source/query/group整组抽样，重算 Accuracy/Macro-F1差、CI；如有多轮，保留同item全部repeat一起采样而非把repeat当独立记录。同配对A/B共同抽样。公开effect size和95%CI，而不仅p。

Exact McNemar基于A/B正确与否discordant counts，只对独立单位适用；RAG同query多passage、模板变体不能直接行级McNemar称严谨。可作为标注“独立假设不成立”的辅助，最好改用组级paired permutation/sign-flip；或预先规定每group抽一条，降低统计力但满足所述条件。不要简单将group内majority correctness当原任务分类结果而不定义新的estimand。

先冻结重点比较（建议每scene JEV vs Gemini、JEV vs dev选定最佳传统流水线，6个确认性比较），Holm校正这6个；如果正式比较全部7模式pair=21×3=63，也须纳入校正。不能看test选best传统再用未校正检验。其余分析exploratory标记。

“没显著差异”不等于两模型一样；不能用未显著做等效。若研究目标等效/非劣，先定义业务可接受margin与专门设计，不在结果出来后追加。

## 七、样本量和资源边界

200独立case在Accuracy约90%时，简单正态95%半宽约4.2个百分点；最保守50%约6.9点。cluster相关会更宽。100%/200仍不能说明绝不出错，简单rule-of-three未观测错误率上界约1.5%（仍只适用独立同分布近似）。

检测配对accuracy差需要看discordance。粗略近似n≈(1.96+0.84)^2×d/Δ²（双侧5%、80%power，d为两系统输出正确性不同的比例）；若d=0.10：5点差约314独立case，3点差约872；多重校正和cluster进一步增加需求。应把这些当设计近似，不能从dev小样本硬算一个保证显著的n。严谨实验允许得出“证据不足以区分”，不能以追求有说服力为由一直扩样本直到显著。

若三scene各240test且2远程×3round，正式远程至少4320 logical requests；采用RAG200query×2pair则更高。加dev prompt候选、身份探测、warmup、HTTP retry都有额外成本/时间。按现有每请求约1.3s/4.5s仅粗估串行请求运行几小时，网关限速和失效率未知。预算和concurrency先记录；不能为了快提高并发后仅比较不同负载时段延迟。若经费不足，优先保留3主scene独立test规模与一次冻结主实验，降低探索性配置/重复轮；不能静默削减n或只测有利scene。

## 八、执行前的硬门槛（go/no-go）

G1：确定每scene任务语义、gold来源、许可证、歧义处理、自然数据与stress区别；自建标签无人工复核时研究定位降级，明确标注。
G2：每主scene >=200实际test rows，并如实公布独立group数量；无后缀凑数；group/text/document overlap=0，近重审查记录存在。RAG query与passage都有且gold不在payload。
G3：完整冻结manifest/prompt/config/search/analysis/code hashes，test不参与开发选择。
G4：LogReg/SVC/XGB/LGB train/dev合理搜索完成、逐candidate可审计；RAG交互特征验收和基本sanity check通过。
G5：failure注入和resume测试通过：HTTP429后retry、拒答、非法json、非法类别、timeout、单条失败继续、进程重启不丢已记记录，不额外免费假设；metrics left-join manifest，计划数不变。
G6：非付费mock上的信息一致性与gold隔离检查通过；真实dev小规模canary用于接入核验，canary不能进入test。版本不可确认时scope降为网关服务，不伪造。
G7：预注册主比较、CI、multiplicity、repeat解释、成本unknown处理，且预算足以覆盖freeze planned jobs。任何变更写amendment并保留原因。

## 九、完成后应当和不应当写什么

可以写：在某冻结版本数据/语言/标签/网关/时间条件下，某系统的all-attempt表现为X，和B的配对差为Δ，95%group CI为[…]；错误和失败分别是什么；哪些来源/边界例更难；传统流水线训练资源和零样本远程条件分别是什么。

不得写：JEV全面优于LLM/传统ML；XGB/LGB不适合文本；合成满分=真实业务满分；片段相关满分=RAG正确率满分；路由分类准=Agent任务完成率高；无显著=模型一样；重复HTTP调用是新增独立样本；用API别名证明官方底层版本；LLM评审等于人工真实gold；缺失usage=零成本；成功部分全对=所有请求完成。

实验完整的定义是：planned样本、模型、重复、日志、统计和数据质量各有可核实交付；不是凑出强结论。最有说服力的结果也可以是“当前样本和服务条件下，JEV与Gemini的差异尚不确定”。
