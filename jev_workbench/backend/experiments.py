"""Built-in experiments and deterministic simulation policies.

The built-in datasets are generated from fixed templates instead of being
sampled at runtime.  This keeps every run reproducible (seed 42 in the UI),
while giving the workbench enough cases to make accuracy and latency
comparisons meaningful.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple


def _balanced_records(
    prefix: str,
    label_templates: Sequence[Tuple[str, Sequence[str]]],
    suffixes: Sequence[str],
    fields: Sequence[str],
) -> List[Dict[str, Any]]:
    """Expand fixed templates into balanced, deterministic records.

    Each label gets one record for every template/suffix pair.  The helper is
    deliberately small and data-only: it does not use randomness or external
    services, so the catalog is stable across machines and test runs.
    """
    records: List[Dict[str, Any]] = []
    index = 1
    for label, templates in label_templates:
        for template_index, template in enumerate(templates, start=1):
            for suffix in suffixes:
                values = {field: template for field in fields}
                values[fields[-1]] = f"{template}{suffix}"
                values["id"] = f"{prefix}-{index:03d}"
                values["label"] = label
                # Suffix variants of one core sentence stay in one split.
                values["group_id"] = f"{prefix}-{label}-{template_index:02d}"
                records.append(values)
                index += 1
    return records


_INTENT_SUFFIXES = (
    "请尽快帮忙看看。",
    "麻烦告知下一步怎么做。",
    "这是今天上午发生的。",
    "我可以提供订单号。",
    "希望今天能收到回复。",
    "请确认是否需要补充材料。",
)

_INTENT_RECORDS = _balanced_records(
    "intent",
    (
        (
            "billing",
            (
                "我刚刚发现信用卡被重复扣款了，能帮我核查吗？",
                "订单已经取消，但款项还没有退回。",
                "这笔支付显示成功，余额却被扣了两次。",
                "我想申请本月订阅费用的退款。",
                "发票金额和实际支付金额不一致。",
                "付款时出现失败提示，但银行卡已经扣款。",
                "请问企业账户的账单在哪里下载？",
                "自动续费我没有开启，为什么产生了扣款？",
                "退款申请提交后，什么时候能到账？",
                "优惠券没有生效，订单金额多收了。",
                "能否把这笔订单改成其他支付方式？",
                "充值金额没有到账，请检查交易记录。",
            ),
        ),
        (
            "technical",
            (
                "登录后页面一直显示 500 错误，怎么处理？",
                "上传文件时浏览器卡住，最后没有完成。",
                "手机端打开应用会闪退。",
                "我修改密码后无法再次登录。",
                "接口请求经常返回超时。",
                "页面加载到一半就白屏了。",
                "验证码一直收不到，无法验证身份。",
                "导出报表时程序提示系统故障。",
                "更新客户端后通知功能失效。",
                "API 调用返回 429，请问怎样重试？",
                "搜索功能输入关键词后没有结果。",
                "同步数据时任务一直停在处理中。",
            ),
        ),
        (
            "complaint",
            (
                "客服三天都没有回复，我很失望。",
                "这个问题来回提交两次，至今没有人跟进。",
                "承诺的处理时限已经到了，仍然没有结果。",
                "客服的回复太敷衍了，我想正式投诉。",
                "同一个问题被转接了四次，体验很差。",
                "售后一直推诿责任，请给出明确解释。",
                "我已经提供全部资料，却还要重复说明情况。",
                "服务质量下降得很明显，希望有人重视。",
                "反馈提交一周了，为什么没有任何进展？",
                "这次处理让我很不满意，请安排专人联系。",
                "你们的承诺没有兑现，我需要一个说法。",
                "客服一直让我等待，完全没有解决问题。",
            ),
        ),
    ),
    _INTENT_SUFFIXES,
    ("text",),
)


_RAG_TOPICS: Sequence[Tuple[str, str, str]] = (
    ("如何申请退款", "用户可以在订单页提交退款申请，审核通常需要两个工作日。", "首页新增了深色模式和快捷键设置。"),
    ("API 限流是多少", "公开 API 每分钟最多 60 次请求，超出后返回 429。", "团队头像可以在个人资料页更换。"),
    ("订单如何取消", "付款前可以在订单详情页点击取消订单，已发货订单需要联系客服。", "本季度产品路线图已经发布。"),
    ("怎么修改登录邮箱", "在账户安全设置中验证旧邮箱后即可绑定新邮箱。", "活动页面展示了最新社区文章。"),
    ("如何开启双因素认证", "进入安全中心，扫描验证器二维码并输入一次性验证码。", "客户端支持三种主题颜色。"),
    ("报表在哪里下载", "在数据分析页面选择时间范围后点击导出按钮即可下载 CSV。", "研发团队下周会举办一次线上分享。"),
    ("企业如何开具发票", "管理员可在财务中心填写抬头和税号后申请电子发票。", "新版本更新了导航栏的图标样式。"),
    ("密码忘记了怎么办", "点击登录页的忘记密码，通过绑定手机重置新密码。", "设计系统新增了四种圆角规格。"),
    ("如何查看 API 调用记录", "控制台的开发者日志会按时间展示每次请求的状态码。", "本周社区投票选出了新的活动主题。"),
    ("客服工作时间是什么时候", "在线客服每天 9:00 至 21:00 提供人工服务。", "个人资料页可以上传新的头像。"),
    ("如何升级套餐", "在订阅管理页面选择目标套餐，确认差价后即可升级。", "帮助中心增加了一个产品术语表。"),
    ("数据多久同步一次", "系统每 15 分钟同步一次外部数据，也可以手动触发同步。", "首页轮播图本月更换了三张插画。"),
    ("支持哪些文件格式", "批量导入支持 CSV、XLSX 和 JSON 三种文件格式。", "团队空间可以自定义背景颜色。"),
    ("如何邀请团队成员", "管理员在团队设置中输入成员邮箱并选择角色后发送邀请。", "产品博客发布了本季度的设计回顾。"),
    ("退款多久到账", "审核通过后原路退款通常在 3 至 7 个工作日内到账。", "导航菜单现在支持按字母顺序排列。"),
    ("如何删除项目", "打开项目设置并点击删除，输入项目名称确认后才能完成操作。", "新手引导增加了三个快捷操作示例。"),
    ("怎么设置通知", "在通知偏好中可以分别开关邮件、站内信和移动推送。", "配色面板提供了浅色和深色两套主题。"),
    ("登录为什么需要验证码", "检测到新设备登录时，系统会额外要求短信验证码确认身份。", "本周榜单展示了最受欢迎的模板。"),
    ("如何查看用量", "控制台概览页显示本月调用量和剩余配额。", "帮助中心首页新增了常见问题导航。"),
    ("服务出现故障时怎么办", "请先查看状态页公告，若持续异常可提交故障工单。", "品牌中心提供了最新的字体下载包。"),
)

_RAG_SUFFIXES = ("（网页端）", "（管理员视角）", "（本月遇到的问题）", "（请给出具体步骤）", "（急用）")


def _build_rag_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    index = 1
    for query, relevant, irrelevant in _RAG_TOPICS:
        for suffix in _RAG_SUFFIXES:
            records.append({"id": f"rag-{index:03d}", "query": f"{query}{suffix}", "passage": relevant, "label": "relevant", "group_id": f"rag-topic-{_RAG_TOPICS.index((query, relevant, irrelevant))+1:02d}"})
            index += 1
        for suffix in _RAG_SUFFIXES:
            records.append({"id": f"rag-{index:03d}", "query": f"{query}{suffix}", "passage": irrelevant, "label": "irrelevant", "group_id": f"rag-topic-{_RAG_TOPICS.index((query, relevant, irrelevant))+1:02d}"})
            index += 1
    return records


_RAG_RECORDS = _build_rag_records()


_ROUTING_SUFFIXES = (
    "，请给出可执行的下一步。",
    "，希望今天完成。",
    "，结果需要附上依据。",
    "，请按优先级处理。",
    "，团队马上要用。",
    "，尽量说明你的判断。",
)

_ROUTING_RECORDS = _balanced_records(
    "route",
    (
        (
            "search",
            (
                "查一下本周关于 JEV 的开源项目",
                "搜索最近发布的中文 RAG 评测报告",
                "帮我找资料比较向量数据库",
                "检索社区里关于提示词缓存的讨论",
                "查找这家公司最新的 API 文档",
                "搜索 GitHub 上可用的日志分析工具",
                "帮忙查一下本地部署模型的教程",
                "找一份可信的多模态模型基准资料",
                "检索过去一年 Agent 路由的论文",
                "查找用户反馈中最常见的问题",
                "搜索开源许可证的兼容性说明",
                "帮我整理网上关于 JEV 的使用案例",
            ),
        ),
        (
            "code",
            (
                "修复这段 Python 的异步竞态",
                "请检查这段代码的内存泄漏",
                "为这个接口补一个重试机制",
                "把 SQL 查询优化到可接受的延迟",
                "排查服务启动时报错的原因",
                "给前端组件增加表单校验逻辑",
                "实现一个批量导入 CSV 的脚本",
                "修复并发任务偶发重复执行的 bug",
                "把这段 Java 代码改成线程安全版本",
                "为 API 增加参数校验和错误处理",
                "定位部署后 502 错误并给出修复代码",
                "为实验平台写一个结果导出函数",
            ),
        ),
        (
            "writing",
            (
                "把实验结果改写成一篇博客",
                "写一份面向新人的使用教程",
                "将这段技术说明润色成公众号文章",
                "为发布会准备一篇产品介绍稿",
                "写一个比较两个模型的评测总结",
                "把会议记录整理成结构清晰的文章",
                "为网站首页写一段简洁的产品文案",
                "写一封向用户解释延迟问题的邮件",
                "将研究结论改写为易懂的科普文章",
                "为这组数据写一份分析报告",
                "帮我拟一个 JEV 实验的标题和摘要",
                "把开发日志写成周报",
            ),
        ),
    ),
    _ROUTING_SUFFIXES,
    ("task",),
)


_EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "id": "intent-classification",
        "name": "中文意图分类",
        "description": "把日常客服消息分到 billing、technical 或 complaint。",
        "task": "classification",
        "labels": ["billing", "technical", "complaint"],
        "records": _INTENT_RECORDS,
    },
    {
        "id": "rag-relevance",
        "name": "RAG 片段相关性",
        "description": "判断检索到的知识片段是否能回答当前问题。",
        "task": "relevance",
        "labels": ["relevant", "irrelevant"],
        "records": _RAG_RECORDS,
    },
    {
        "id": "agent-routing",
        "name": "Agent 路由选择",
        "description": "为一条任务选择最合适的处理路径。",
        "task": "routing",
        "labels": ["search", "code", "writing"],
        "records": _ROUTING_RECORDS,
    },
]


def list_experiments() -> List[Dict[str, Any]]:
    """Return public catalog metadata, including sample records for the UI."""
    # Return a copy so callers cannot mutate the process-wide catalog.
    return [{**item, "labels": list(item["labels"]), "records": [dict(record) for record in item["records"]]} for item in _EXPERIMENTS]


def get_experiment(experiment_id: str) -> Dict[str, Any]:
    for item in _EXPERIMENTS:
        if item["id"] == experiment_id:
            return {**item, "labels": list(item["labels"]), "records": [dict(record) for record in item["records"]]}
    raise KeyError(experiment_id)


def split_records(records: List[Dict[str, Any]], seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """Split suffix variants by core-template group, keeping labels present.

    The built-in datasets deliberately expose ``group_id``.  A random row
    split would put near-identical suffix variants in both train and test;
    assigning whole groups avoids that leakage while remaining deterministic.
    """
    del seed  # The catalog is deterministic; the argument documents the contract.
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("group_id", record["id"]))].append(record)
    grouped = list(groups.items())
    labels_by_group = {key: {str(item.get("label", "")) for item in values} for key, values in grouped}
    if any(len(labels) > 1 for labels in labels_by_group.values()):
        ordered_groups = sorted(grouped)
        assignments = _split_group_keys([key for key, _ in ordered_groups])
    else:
        per_label: Dict[str, List[str]] = defaultdict(list)
        for key, labels in labels_by_group.items():
            per_label[next(iter(labels))].append(key)
        assignments = {"train": [], "validation": [], "test": []}
        for label in sorted(per_label):
            allocation = _split_group_keys(sorted(per_label[label]))
            for split_name in assignments:
                assignments[split_name].extend(allocation[split_name])
    return {split: [record for key in assignments[split] for record in groups[key]] for split in assignments}


def _split_group_keys(keys: List[str]) -> Dict[str, List[str]]:
    total = len(keys)
    train_n = max(1, int(total * 0.6))
    validation_n = max(1, int(total * 0.2))
    if train_n + validation_n >= total:
        validation_n = max(1, total - train_n - 1)
    return {
        "train": keys[:train_n],
        "validation": keys[train_n:train_n + validation_n],
        "test": keys[train_n + validation_n:],
    }


def simulate_prediction(experiment: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic baseline policy used for the local simulation mode."""
    exp_id = experiment["id"]
    if exp_id == "intent-classification":
        text = record["text"].lower()
        scores = {
            "billing": 0.85 if any(token in text for token in ("扣款", "退款", "支付", "账单", "订单", "款项", "发票", "银行卡", "续费", "到账", "优惠券", "充值")) else 0.08,
            "technical": 0.85 if any(token in text for token in ("错误", "登录", "故障", "500", "崩溃", "上传", "卡住", "闪退", "密码", "超时", "白屏", "验证码", "导出", "通知", "api", "429", "搜索", "同步")) else 0.08,
            "complaint": 0.85 if any(token in text for token in ("客服", "失望", "投诉", "没有回复", "不满", "推诿", "重复", "承诺", "反馈", "专人", "敷衍", "等待")) else 0.08,
        }
    elif exp_id == "rag-relevance":
        query = record["query"].lower()
        passage = record["passage"].lower()
        # Chinese text is not whitespace-tokenized.  Match a fixed set of
        # domain terms so every generated relevant pair is recognized while
        # unrelated passages remain negative examples.
        terms = ("退款", "api", "限流", "订单", "邮箱", "认证", "报表", "发票", "密码", "日志", "客服", "套餐", "同步", "文件", "邀请", "到账", "通知", "验证码", "用量", "故障")
        relevant = any(term in query and term in passage for term in terms)
        scores = {"relevant": 0.9 if relevant else 0.12, "irrelevant": 0.1 if relevant else 0.88}
    else:
        text = record["task"].lower()
        scores = {
            "search": 0.88 if any(token in text for token in ("查", "搜索", "开源", "资料", "检索", "找", "教程", "文档", "案例")) else 0.06,
            "code": 0.88 if any(token in text for token in ("代码", "python", "修复", "bug", "异步", "接口", "sql", "内存", "脚本", "java", "参数", "校验", "502", "导出", "实现")) else 0.06,
            "writing": 0.88 if any(token in text for token in ("博客", "改写", "文章", "写", "教程", "润色", "公众号", "介绍稿", "总结", "记录", "文案", "邮件", "科普", "分析", "摘要", "周报")) else 0.06,
        }
    prediction = max(scores, key=scores.get)
    confidence = float(scores[prediction])
    return {"prediction": prediction, "confidence": confidence, "probabilities": scores}
