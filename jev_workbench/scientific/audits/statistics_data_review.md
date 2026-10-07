# statistics.py / datasets.py 只读审查

2026-10-07。仅离线审查与最小fixture执行，不修改root文件，不调用API。现有statistics+datasets测试17项通过（0.67s）。数据真实人工复核、版本canary、预算及正式冻结仍pending。

## 1. 统计上最重要的设计缺口：class抽样与bootstrap不匹配

位置 datasets.py:152-165（capped stratified抽样），statistics.py:100-109（无分层group重采样）、92-97（缺席class=0）。

可复现：60类，每类1条独立group，A/B都全部预测正确。paired_bootstrap(...,iterations=10000,seed=20261007)给macro_f1 estimate=1.0，但a/b ci95=[0.55,0.7166666666666667]，准确率ci=[1,1]。这是resample缺席类别时固定label-set零分导致的统计偏差；区间甚至不含已知全正确的估计值。不会破坏A-B完全相同的零差，但会影响单模型CI，并可能影响模型差CI/稀有类权重。

MASSIVE cap=10不是必然触发这个极端：类在重采样中缺席概率低得多。但exact/near清理后某些类可能只剩1～2条，不能假定仍各10条。主label set全部固定是合理透明的score定义，不应为了修CI简单在每bootstrap丢掉未出现类（那又换estimand）。

建议：记录真实sampling design，在group为单label的MASSIVE capped class设计中，class-stratified group bootstrap保留每类原group数；配对A/B共用分层weights。混合label的RAG query group不能随意把passage按class拆组，需要明确自然group-bootstrap或按query来源/类别模式分层。输出缺席class比例/稀疏group诊断；仅单组时区间不作证据。加60类×1 perfect、稀有class与混合labelquerytests。Protocol预冻结实际bootstrap strata，而不是事后见CI难看再改。

## 2. 近重筛查有真实跨split残余边界

位置 datasets.py:101（只eligible>=min_length）、119-126（短文本只exact）。

最小fixture：test文本“退款退款退款”（6字），train“退款退款退款退”（7字），各独立group，其他字段合法。用同款char TF-IDF(2,4)其cosine=0.9735151541830581，大于默认0.90；clean_near_duplicates(...threshold=.90,min_length=8)仍保留2条，因为均不eligible且不exact。

这不是semantic paraphrase边界，而是算法自己定义的相似度阈值下仍有被跳过的crosssplit近重。audit目前只报min_length，没有skipped count/residual验证；不能仅把near_duplicate_audit设“完成”就宣称所有话语无近似交叉。

建议短文本另行筛查或降低min_length、人工审查短文本候选，明确短语误报处理；报告short_skip_count、source/class/split删前删后计数。清理后独立重新计算crosssplit threshold残余，不只复用已删除pairs计数。semantic paraphrase和预训练污染仍不可排除，应保留边界。

算法其他部分检查：units按test/val/train排序，screens覆盖所有非traineligible；greedy保留winner、删除later，使保留的>=min_length集合中任何nontrain对latergroup的above-threshold lexical edge都应被消除；train/train近重不扫描（除exact）。这个范围要如实说明，非跨split全局语义去重。

## 3. structural_ready允许真正无效的输入

位置 datasets.py:59-62。

最小fixture：单test row的input={'text':None}，label='a'、group非空，min_test_groups=1。audit_dataset给structural_ready=True、errors=[]。原因str(None)变'None'；数字、bool、list同样可能通过。remote.build_payload明确要求实际非空str，会在执行前拒绝。

建议audit和payload一致，所有白名单字段必须isinstance(v,str)且strip非空；group/id/labels也明确类型。structural_ready保持“结构门槛通过”，ready仍False直到许可/近重/gold/预算等完整验收，不要让统计数替代这些。

## 4. exact冲突删除与near整query删除是不同政策

位置 datasets.py:35-37与127-129。

near按query/textunit删除整个group，符合RAG独立单位约束，现有测试验证了这一点。

exact仅按完整input（query+passage）建立family；遇同输入不同gold，删除冲突pair的所有副本，留下同query其他pair。fixture：test q0=(Q,P0,'a'), test q1=(Q,P1,'b')，train q2=(Q,P0,'b')。exact后只剩q1，q这个testgroup不再包含原相关/不相关构成。后续near不会恢复已删pair，也未必删除此query。

不能把exact称“纯input-only”：它读取gold决定conflict整family剔除。这个操作并非一定错误，预先定义的标注冲突隔离是合理质量政策，但往往排除歧义/困难例，可能提升表观分数，而且改变query内passage比例和class/source分布。

建议先冻结冲突政策：整query隔离、只隔离pair但记录query构成变更，或交由真实标注审查后保留。记录所有raw IDs与删前后各class/query统计、冲突率；如果可行额外报告包含conflict的敏感性分析（没有可靠gold时不能凭模型输出修标签）。sample metadata明确“使用官方标签的冲突清理”，不称label-blind。官方test优先本身不会因为查看模型输出造成挑分，但清理后是官方test派生子集，不能称原完整官方分布。

## 5. 统计核心实现：已确认正确与需写清的范围

- failure-aware score矩阵包含NO_VALID_OUTPUT，失败计gold类FN；valid-only另单独去掉failure列，分母透明。TP/precision/recall/F1计算正确。
- macro_policy固定suppliedlabelset，unsupportedclasses=0已显式输出。若某类在自然test无support，需要正文说明不是该类实测为0；最好列absentclasses诊断，禁止在结果出来后删除难类。valid-only里某class全拒答因此absence，依fixed政策仍0，不能简单称“有效输出全部正确所以Macro-F1=100%”。
- paired_bootstrap按group一次multinomial共同A/Bweights，保留同query全部passage；不把repeat增加n。_validate拒绝重复item_id，要求先filtermodel/repeat；pairgold/group必须一致。
- bootstrap与主accuracy是row-weighted/passage-weighted，不是query平均accuracy。group重采样保证相关性处理，并不改变点估计weights。querypassage数不同则大query权重更高。协议须冻结这是目标（片段总体），或加equal-query平均指标与对应group-leveltest；不能把pair准确率称“独立query成功率”。
- perfectclassifier的非参数bootstrap会退化[1,1]，不能解释为总体准确率一定100%。独立一对一case可补Wilson/Clopper-Pearson单比例CI；cluster数据不能无声明使用rowbinomial。差异也可能因无观察discordance退化0；需明确条件经验区间和样本有限，而非保证等价。
- mcnemar_exact确认每group仅1row并用binomtest两侧tail，正确拒绝相关passage。只给itemIDs唯一无法证明实际独立，应依数据来源group审计。
- paired_cluster_test使用group内correctnesssum再整体signflip，保留group；精确或MonteCarlo+1计算合理。其p检验metric仅all-plannedaccuracy，未检验Macro-F1。解释依赖独立group、A/B可交换/差值对称的置换零假设，不能笼统声称任意非正态分布都有效。
- Holm排序/monotone/rereturn原顺序算法正确，但只是接收裸p列表，没有组织confirmationfamily的实现。Protocol当前说accuracy、Macro-F1、validrate均主指标，随后只说JEV–Gemini/JEV–bestlocal做Holm；需要指定确认性显著性检验只针对accuracy（2×正式task数量），其他指标给CI和描述，或把要做显著性推断的所有metrics加入family并有相应test。不能拿accuracy的adjustedp给F1或validrate盖章。

## 6. 测试覆盖与准备度

现有17tests验证主数学、pairing/group不膨胀、partial分母、exact/near基本删除与结构ready未冒充实验ready，属于有效基础测试。尚未覆盖上述稀有classboot设计、短文本近重残余、Noneinput、RAG冲突partialquery、perquery权重政策以及Holm确认性family。

load_massive test_cap_per_label=0/负数没有validation；0会清空test但audit通常发现n不足，负数切片会静默“去掉最后若干条”。正式CLI应只接受None或positiveint（排除bool），min_test_groups也positive。

本审查不建议为了展示可信而强求某model显著胜出。上述门槛满足后，即使差CI跨0或过滤/成本范围不确定，仍是可以清楚交付的科学结果。

## 7. 修复后二次复核（2026-10-07，当前候选数据）

已确认：

- paired_bootstrap新增stratify='gold_label'，按单label组保持各类实际分配，拒绝混合labelgroup；60类各1条perfectfixture恢复Macro-F1 estimate=1、ci=[1,1]，同时rare_strata=60并输出条件经验区间警示。先前因缺类导致CI0.55～0.72的实现问题已修复。这个[1,1]仍非总体百分百准确证明。
- audit_dataset现在拒绝None/numeric/bool非字符串输入，原Nonefixture不再structural_ready。
- 近重默认min_length=2，short_units_exact_only显式记录；先前6/7字相似例进入screen。PROTOCOL改成预定词面筛查，不声称语义泄漏全部排除。
- Protocol固定确认性Accuracy指标及2×正式task数的Holm family，Macro-F1和其他仅探索性CI；解释分层MASSIVE/整queryRAGbootstrap与固定60label、支持59label的两种F1。

独立重建massive source -> exactclean -> cappedtest，使用原near审计输入集合fitTF-IDF，再对最终保留数据跨split检查：cosine≥0.90的残余pair数=0（双方长度≥2的所述词面basis）。没有把重新fit保留集合后变化的IDF当同一预定算法；没有对语义复述或预训练记忆作保证。

massive_candidate.json records hash和audit.manifest_sha256完全一致（eaa5b3d8f38db461d6e5b43d22479910ec0f798b2615a03afc163461464a8b36）。train=10684/60类，validation=1958/59类，test=558/59类；test558个record/group，结构验收通过，ready仍False。

冻结前保留caveats：

1. test缺cooking_query，因此固定60类Macro-F1满分上限是59/60≈98.33%（即其他所有类全对）；59supported类F1另报但不能删掉这个候选类别。validation缺audio_volume_other，不能声称该类已充分dev选优。train cooking_query仅4条、validation2条。
2. test稀有类general_greet1条、iot_hue_lighton3条、music_dislikeness4条，虽总558大于200，不能推断各60类均充分验证；分层CI需报告真实stratum_support和rare_strata。
3. sampling.selected_test_records=559是near清理前，最终558；正文/预算manifest以最终records为准。分层cap改变官方自然频率，不能称原完整官方test分布。sampling中“input-only exact cleaning”还需改成含标签冲突隔离的真实口径。
4. 词面near另删除25groups/records，短于2字的8个unit仅exact检查；相关歧义剔除率、class/source删前后分布要与冻结数据同时归档。
5. 原始utteranceID单独group能消除同话语重复计数，但没有建立speaker/session层级独立性；泛化对象是此来源的独立话语近似，而非用户群体或真实生产流量。
6. MASSIVE官方human/localization标签沿用，并未在本项目全量真实双人标注；路由真人gold与RAGsource许可/标注质量需独立门槛，不能由MASSIVE ready替代。
7. n=558也无法保证精确区分所有模型的小差异；确认性comparisons在dev选bestlocal后冻结，不能看test选winner，且不追逐显著。
