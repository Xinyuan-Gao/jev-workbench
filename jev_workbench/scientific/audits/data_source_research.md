# JEV 严谨重做：数据源核验（2026-10-07）

本报告只做数据与标注核验，未请求付费模型，未改实验平台。网页读取按 agent-reach 网页/GitHub 渠道进行；HF 主站超时，数据文件通过 hf-mirror 下载，文件身份以官方数据 revision、LFS SHA256 和原文件结构核验。agent-reach CLI 本机未安装，无法运行 check-update。

## 1. MASSIVE 1.1 zh-CN：推荐中文意图主任务

官方仓库：https://github.com/alexa/massive ，读取时 repo commit f966f21846043aabef9b0f974fa7970027f43738。
官方数据：https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz 。本次直接从官方 S3 完整下载 40,251,390-byte 1.1 归档，提取 zh-CN.jsonl 和 LICENSE。
数据许可 CC BY 4.0；工具代码 Apache 2.0。两者不能混写。

实际统计：16,521 条；train 11,514、dev 2,033、test 2,974；train 包含 60 个 intent，dev/test 各 59；18 个 scenario。JSONL 含 id、locale、partition、scenario、intent、utt、annot_utt、worker_id、slot_method、judgments。模型输入仅 utt 和预先固定的类别说明，不能带 intent/annot_utt/judgments。

来源是 SLURP 英文语音助手单轮指令，由 MTurk 工作者本地化到中文，带多个人工意图一致性、语法、拼写审核记录。它有人工作业依据，但不是自然产生的中文客服消息，结论范围应是中文语音助手意图分类。

本地文件：1.1/data/zh-CN.jsonl；SHA256 992bf0bef3d678f08c27e514739bc851163e8f40f530bfb4d5970a2c24408ace。

科学限制：2022 年公开基准，JEV/Gemini 可能在预训练或后续训练看过，无法证明无污染；中文本地化可能有表达局限；稀有类支持不足。test 缺 cooking_query，全目录 cooking_query 只有 6 条（train4/dev2）。test 的 general_greet 只有1条。不能宣称60类各10条。

按固定 seed42、按 intent/numeric ID 排序、每 test 类 min(10,n) 无放回抽样，实际 560 条/59类；原59类别test支持与选中 IDs 在 massive_audit.json。测试可保留60候选，但macro-F1统计必须明确 absent class 口径，另外列各类support，不能悄悄把不存在类别算成失败或隐去。

原始目录有902条重复文本，53种文本对应不止一个intent。跨分区：train-dev 132种重复文本/144条dev记录，train-test187种/223条test记录，dev-test46种/55条test记录。560选中test中43条与train重复、12条与dev重复，对应需排除train97条source记录与dev10条source记录。

近重检测仅供审计：NFKC、小写、去Unicode标点/空白，char2–4gram TF-IDF cosine>=0.90且双方长度>=8，检测器拟合全文只做词面候选检索，不能用于训练模型特征。560sample对应train102条source候选、dev10条。较高相似句子未必是泄漏，必须预先定义保守规则。非完全重复示例见 massive_audit.json 的 nonexact_examples；不把LLM复核称人工复核。

推荐：固定官方test抽样或完整test，先冻结 IDs，再清理训练/验证完全重复；近重做预先确定的敏感性分析或保守词面清理，记录所有排除。训练特征仅 fit train；dev 调参；test 不改prompt。高词面近似被移除后训练数据和类别支持也要审计。

## 2. T2Ranking 原始：推荐问题—片段信息相关性主任务

官方仓库：https://github.com/THUIR/T2Ranking ，repo commit 3ab0a0de72dd50bf84d852a985f6188334781403。
官方数据：https://huggingface.co/datasets/THUIR/T2Ranking ，dataset revision 2a369a430a70979223f1b9a41b1919774d46b432。
论文：https://arxiv.org/abs/2304.03679 。数据许可 Apache 2.0（官方README、HF card、论文第一页脚注一致）。

原始来源：搜狗真实搜索日志中的问题型query，多搜索引擎取得网页段落，专家标注。论文说明至少三人标注，不一致时第四人，全部不一致丢弃，最终多数票；该描述来自论文，原TSV只给最终grade，并非每人原标签。

等级：0完整不匹配；1话题相关但不满足信息需求；2部分满足；3精确包含答案。官方retrieval二分类口径是grade>=2 relevant、grade0/1 irrelevant。因此适合“是否能为回答提供信息”；不可把grade1偷偷当relevant。

官方README报告 train258042query、dev24832、test24832；实际文件去header后train258042/dev24831/test24831。公开qrels.train有1613421pair，qrels.dev有400535pair、24827query；dev有4query无qrel（4491,10705,21775,23509）。没有公开qrels.test，不能宣称获得官方test成绩。

dev grades：0=217692,1=63911,2=94364,3=24568；train：0=699080,1=169679,2=415994,3=328668。

queries train/dev/test 文本 exact 均无交集；NFKC去标点后的train-dev交集也0。不同split qid从0重新编号，ID必须带split前缀。train/dev共享7389 passage IDs，在按query划分的任务中不等于整条query-passage泄漏，但要说明。

原collection.tsv为pid\ttext，总3,659,243,528 bytes；LFS SHA256 07b84e543e9ba696124a727c00629d5bce586631648c436c98dd6e9b146da212。已完整下载并校验该SHA256，扫描得到2,303,643条片段。collection_sample.tsv仅保留早期512KiB可行性取样，不是最终source。BM25/DPR候选文件只有qid/pid/rank/score，没有text。

按seed42从dev同时有正负的22720query预抽200query，涉及3136pair/3133pid；首512KiB只命中2pid（0.064%），不能拿前面片段替代完整随机样本。选中query只是可行性示例，尚未正式数据合同冻结。选择200独立query而非把同一query的200pair当200独立样本；置信区间/bootstrap以query为cluster。

推荐：原train按query划分训练/验证、dev预抽冻结为本实验评估集，明确原数据split叫dev、在本实验才是heldout评估。模型二分类采用官方阈值，所有四级label保留用于分层错例。原始文件和SHA在t2_query_audit.json。目前已完成完整流扫描和原pid抽取。正式source采用统一seed20261007，从22720个有正负grade的原devquery按numericqid排序后random.sample1600；前200test、次200validation、剩1200train，仍为source_preparation未冻结。保留全部26041qrels（train19458/val3324/test3259），25823unique pid全部找到，无空片段/重复文本对/跨fold规范化重复query。选中manifest为t2_source_selection_manifest.json，source_records按fold分别输出t2_source_train/validation/test.jsonl。最终模型输入pair由root在正式请求前固定为每query每原grade至多1条，不补缺grade，不根据模型结果重采。

## 3. C-MTEB/T2Reranking：已下载，但含义不同

官方派生数据：https://huggingface.co/datasets/C-MTEB/T2Reranking ，revision 76631901a18387f85eaa53e5450019b87ad58ef9。
官方任务代码：https://github.com/FlagOpen/FlagEmbedding/blob/master/research/C_MTEB/C_MTEB/tasks/Reranking.py 。评价任务沿用T2Ranking论文，但dataset card只写缺资料，未提供独立license或转换描述；license可引用上游T2Ranking Apache2，并明确派生卡未单列许可。当前mteb task元数据也将license标为not specified。

完整parquet：t2reranking-dev.parquet，120,293,598 bytes；SHA256 ceaf1ba1a4777dbb7aecad86fb8aa9cae7a678af6d711d5529c238e89bb6adff，与官方LFS哈希一致。pyarrow读取6129 unique query，每行 query、positive[]、negative[]，共100000pairs：45551positive/54449negative。只有dev，没有官方train/test。

逐query核验：用literal TSV可全部6129query映射原dev qid（默认CSV引号解析会导致4处不匹配）；6128完整query的positive数等于原grade1+2+3，negative数等于grade0。最后qid6174被100000pair边界截断，只剩2positive，原本4positive/11negative。

进一步将各side按原qrels同side顺序配pid，与原collection首512KiB已取得的287pair逐文本比对，无一不匹配。其中grade1=51个明确在positive里、grade2=58、grade3=19、grade0=159。这样独立证实grade1进positive，不是只靠数量猜测。之后完整collection下载成功，已对全部100000派生pair逐一比对原pid对应文本，100000全部匹配，0不匹配、0同query同passage冲突原grade。完整原grade分布为negative:grade0=54449；positive:grade1=16107、grade2=23308、grade3=6136。这完整核实派生positive包含grade1，不再只是位置或数量推测。完整审计t2reranking_full_original_audit.json。仍未找到官方公开转换源码；官方HF提交记录只有parquet/README上传。

可用方案：明示任务是“任何非零等级的话题相关性”，使用公开positive/negative，不称“是否足以帮助回答”；按query固定随机分内部train/val/test并报告派生split，不声称官方test。本轮已流扫描原collection完成核验，并采用原qrels>=2任务，不用派生positive替代gold。整个派生候选涉及pid最大2087702。

风险：此派生数据是原dev前100000pair/前6174附近query，非代表性随机样本；末尾截断；221query只有单侧类别；类别分布不同于原信息满足阈值。只有被明确界定的任务才可比较。

## 4. Agent首步路由：公开真实请求能找到，但无对应标准gold

WildChat官方：https://huggingface.co/datasets/allenai/WildChat-1M ，revision 7d6490e462285cf85d91eabea0f9a954fbddcd1f；论文https://arxiv.org/abs/2405.01470；许可ODC-BY。本版非有毒conversation过滤后的metadata实际837989条，只有train，14个parquet shard，总下载约3.36GB，不是名称中的一百万全部保留。来自真人与GPT3.5/4对话，可筛中文首条user输入；card和schema已取得，暂未下载大shard。

字段含conversation_hash、timestamp、conversation[]和language，但没有search/code/writing人工路由。只提取任务必要文本、sourceconversationhash与源revision，避免向模型送assistant答案和不必要的IP/地域/header信息。同conversation必须同partition，多轮缺前文的首步样本要补齐上下文或列为不可判断。

路由取决于可用工具、当前信息是否足够、是“第一步”还是“最终交付”：写代码前可能要搜索，写文章前可能要查资料；三类不是自然互斥。先定义固定能力与优先级，允许clarify/other或标可接受多标签。两位真实人工独立标注，记录一致率/Cohen kappa和争议裁决，再冻结gold。Subagent标签可作建议，不能写成双人人工gold。路由未通过这个步骤前只做探索，不纳入确认性总排名。

英文补充来源：Databricks Dolly15k，官方https://huggingface.co/datasets/databricks/databricks-dolly-15k；revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a；许可CC BY-SA3.0。原文件dolly.jsonl已下载，15011条，原生8种instruction类别：open_qa3742/general_qa2191/classification2136/closed_qa1773/brainstorming1766/information_extraction1506/summarization1188/creative_writing709。来源是Databricks员工，明示不得用生成AI创作，允许特定类别引用Wikipedia；有真实人工来源但非自然业务日志，只有一个整体文件没有官方train/val/test。可做独立英文“指令类型分类”原生标签对照，不能把openQA强映射成search、creativewriting泛化成唯一writing路由，也不提供充分code类别。公开指令和分类标签同样有预训练污染风险。

## 可复查产物

所有文件位于 /tmp/jev_scientific_research 。
- massive_readme.md/massive_notice.md、1.1/LICENSE、massive_audit.json、audit_massive.py
- t2_readme.md、t2_paper.pdf/txt、t2_api.json、t2_filemetadata.json、t2_loader.py、t2_dataset_factory.py
- queries.train/dev/test.tsv、qrels.train/dev.tsv、t2_query_audit.json、t2_200query_sample.json、collection_sample.tsv（仅不完整前缀）
- t2reranking-dev.parquet、t2reranking_api.json/card.md/filemetadata.json/commits.json、t2reranking_audit.json、t2reranking_positional_audit.json
- wildchat_api.json/card.md；dolly_api.json/card.md/readme.md、dolly.jsonl

统计字段以实际解析记录数为准，保留原来源报告数和差异。正式选样、清理、标签映射与成本范围必须在第一次模型test请求前冻结；不能因为看到分数不好再改任务阈值、抽样或排除规则。

## 最终source交付状态

原collection完整SHA256已验收，根路径 /tmp/jev_scientific_research/collection.tsv。统一seed20261007选中1600query的source准备已完成，未冻结、未测试API。t2_selected_source_records.jsonl总63,292,014bytes，SHA256 cc733d6a1235bf0966108091116bc41285c678f9aab8e6a57f1f8e05d0b6f4f7。每条含真实原query、passage、source qid/pid、human_grade、>=2 binary label与文本对SHA256。注意这些source含gold，需要平台严格分开可送模型输入和评估标签。原片段不截断；source test3259pair有5条超过16384字符、max123841；train最高176215、validation最高41468。正式数据合同要固定完整输入/长度上限与失败处理，不能暗中截断再用完整片段gold评分。
