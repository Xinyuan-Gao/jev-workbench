# JEV 科学对比实验 v2 Implementation Plan

**Goal:** 执行有可信标注、独立测试集、合理本地基线、可追溯远程日志和成对统计区间的新一轮实验。

**Architecture:** 新建 scientific 模块，保留原工作台与历史记录。dataset manifest 管理样本和独立单位；模型输入构建器与 gold storage 分离；逐条 journal 可恢复；统计端从完整 planned manifest 评分，前端读取脱敏进度与结果。

**Tech Stack:** Python、numpy/scipy/scikit-learn、XGBoost/LightGBM、HTTP JSON、原生 HTML/JS。

## 任务及验收

- [x] 数据出处核查：研究代理核实中文/公开候选的原始文件、license、人工标注与官方 split；下载记录和引用保存于 scientific/data_sources。
- [x] 数据合同：写真实行为测试，确保字段白名单、官方 split、独立 group、跨分区重复检测和 >=200 独立 test gate；通过后实现 scientific/datasets.py。
- [x] 统计：先验证失败计入全提交分母、Macro-F1 的 gold 漏报、query 组 bootstrap、paired 差值方向和 Holm 校正；实现 scientific/statistics.py。
- [x] 执行 journal：先用本地 HTTP 替身验证断点恢复、单条拒答继续、原始响应与逐次重试保存、密钥脱敏和 planned 记录完整；实现 scientific/remote.py、scientific/journal.py。
- [x] 合理本地模型：实现 sparse LR/SVM 与 train-only SVD 的树模型；相关性加入 query/passage 特征交互。固定候选网格并保存全部 dev 选择记录。验证 test 从不传入 fit/tune。
- [ ] 正式协议冻结：收集数据审查与模型 dev 结果，确定实际样本量/重复/预算；用 scientific/freeze.py 生成配置、文件 SHA256 和冻结时间。门槛未过不得开始 test。
- [ ] 运行新矩阵：所有模型处理同一份 test，远程至少3轮、本地随机模型至少5seed；逐条落盘。无语义改写重试、无 test 调参、无挑选轮次。
- [x] 科学结果页：在原服务提供 scientific.html，只展示脱敏进度、planned/valid/failed/remaining、来源/版本与冻结信息。刷新和进程恢复验证。
- [ ] 独立审计：研究协议代理审阅当前冻结与实际日志；统计从 journal 重算，验证 terminal 完整、原始结果一致、gold泄漏无命中、区间分母正确。
- [ ] 图文交付：真实截图与来源对应，数据图由冻结结果生成；重写飞书文章为“研究问题→方法→结果→错例→边界”，保留原版备份。

阶段 A 先冻结并完整运行有公开人工 gold 的 MASSIVE/T2 两任务；路由保留为独立阶段 B，待真人标注与裁决。全部结果完整且通过独立重算前不发布新排名。用户已授权全部测试，阶段 A 采用固定全局 HTTP 上限；单价未知则费用记未知，不能以次数上限承诺人民币金额。

## 最终补充验收

- [x] 数据记录从持久原始副本重建，与候选清单记录哈希相同。
- [x] 5pp设计敏感性计算与独立三项枚举交叉验证，不保证足够功效。
- [x] 18次真实验证接入记录与新原始响应复算合同一致，不视为正式成绩。
- [ ] 最终分析器交叉复核、真实请求/响应/终态一致、自己实际源码冻结。
- [ ] 最终图表导出器通过完整性测试和真实渲染人工检查，全部图标注分母与来源。
- [ ] 全科学套件通过后冻结阶段 A，正式矩阵23,408组合、远程7,392首尝试。
- [ ] 正式完整矩阵独立重算与二次审计，才更新博客结论。
