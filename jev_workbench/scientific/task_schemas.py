"""Fixed natural-language label definitions, authored from label taxonomy/train.

No test predictions, gold fields, example IDs or annotations enter a payload.
The task comparison is supervised local pipelines versus pretrained zero-shot
services, not equal training data or equal compute.
"""

INTENT_DEFINITIONS = {
    'alarm_query':'查询已经设置的闹钟。',
    'alarm_remove':'取消、删除或停止闹钟。',
    'alarm_set':'新设闹钟或指定叫醒时间。',
    'audio_volume_down':'降低播放或设备音量。',
    'audio_volume_mute':'静音或停止设备声音。',
    'audio_volume_other':'设置具体音量或笼统调整音量，未表达单纯调高、调低或静音。',
    'audio_volume_up':'提高播放或设备音量。',
    'calendar_query':'查询日历、日程、提醒或已安排活动。',
    'calendar_remove':'删除或取消日历事项、日程或提醒。',
    'calendar_set':'添加、修改日历事项、日程、预约或提醒。',
    'cooking_query':'询问做什么菜或泛指烹饪建议，未具体询问某菜的做法。',
    'cooking_recipe':'询问具体食物的做法、配方、材料或烹饪时间。',
    'datetime_convert':'换算两个地区、时区或时间标准之间的时间。',
    'datetime_query':'查询当前或指定地点的日期、时间。',
    'email_addcontact':'创建或添加邮件联系人。',
    'email_query':'查询邮件、收件箱、未读邮件或邮件内容。',
    'email_querycontact':'查询联系人或其邮箱、电话等联系方式。',
    'email_sendemail':'撰写、发送或回复电子邮件。',
    'general_greet':'问候、打招呼、询问近况。',
    'general_joke':'要求讲笑话或逗人开心。',
    'general_quirky':'其他闲聊、随意表达或不属于其他候选意图的请求。',
    'iot_cleaning':'让智能清洁设备打扫、吸尘或开始清洁。',
    'iot_coffee':'让咖啡机制作咖啡或控制咖啡机。',
    'iot_hue_lightchange':'改变智能灯的颜色、色温、模式或场景。',
    'iot_hue_lightdim':'调暗智能灯。',
    'iot_hue_lightoff':'关闭智能灯。',
    'iot_hue_lighton':'打开智能灯。',
    'iot_hue_lightup':'调亮智能灯。',
    'iot_wemo_off':'关闭智能插座或插座所控制的电源。',
    'iot_wemo_on':'打开智能插座或插座所控制的电源。',
    'lists_createoradd':'创建清单或向清单添加条目。',
    'lists_query':'查询清单或清单上的条目、状态。',
    'lists_remove':'删除清单或从清单删除条目。',
    'music_dislikeness':'表示不喜欢某音乐、歌手、歌曲或音乐类型。',
    'music_likeness':'表示喜欢某音乐、歌手、歌曲或音乐类型。',
    'music_query':'查询歌曲、歌手、专辑、正在播放音乐的信息。',
    'music_settings':'改变音乐播放设置，例如随机播放、循环播放。',
    'news_query':'查询新闻、时事或媒体报道。',
    'play_audiobook':'播放或控制有声书。',
    'play_game':'开始、选择或进行游戏。',
    'play_music':'播放、选择或控制音乐歌曲。',
    'play_podcasts':'播放、选择或控制播客。',
    'play_radio':'播放、选择或控制广播或电台。',
    'qa_currency':'询问货币、汇率或货币兑换。',
    'qa_definition':'询问词语、概念或名称的定义、意思。',
    'qa_factoid':'询问一般事实、人物或百科知识，不属于其他专门问答类别。',
    'qa_maths':'进行数学计算或回答数学问题。',
    'qa_stock':'询问股票、股价或股票市场。',
    'recommendation_events':'寻找、推荐当地活动、演出或可参加的事件。',
    'recommendation_locations':'寻找、推荐地点、商店、餐馆等目的地。',
    'recommendation_movies':'推荐或选择电影。',
    'social_post':'在社交平台发布内容、状态或消息。',
    'social_query':'查询社交平台的消息、动态、热搜或通知。',
    'takeaway_order':'下单订餐或安排外卖配送。',
    'takeaway_query':'查询外卖餐馆、菜品、价格或订单配送状态。',
    'transport_query':'查询交通路线、方向或一般出行方式。',
    'transport_taxi':'叫车、预订出租车或网约车。',
    'transport_ticket':'查询或购买火车、航班、公共交通的票或班次。',
    'transport_traffic':'查询道路拥堵、实时交通或通勤路况。',
    'weather_query':'查询天气、气温、降雨或天气预报。',
}


def intent_schema():
    return {'task':'classification','labels':sorted(INTENT_DEFINITIONS),
            'definitions':dict(INTENT_DEFINITIONS),
            'instructions':'将一条中文语音助手请求归入 MASSIVE 意图类别。只根据输入判断主要意图；不要执行请求。可能缺少上下文时仍选择最符合请求的候选意图。类别名是标识符，含义以中文定义为准。'}


def relevance_schema():
    return {'task':'relevance','labels':['irrelevant','relevant'],
            'definitions':{
                'irrelevant':'片段与问题不匹配，或仅谈论相同话题却没有满足该问题的信息需求（T2Ranking 等级0或1）。',
                'relevant':'片段至少部分满足该问题的信息需求，提供了一部分有用答案或精确答案（T2Ranking 等级2或3）。'},
            'instructions':'判断给定片段是否能为给定问题提供答案信息。必须同时阅读 query 与 passage；只出现相同关键词不足以判相关，不要求片段独自完整回答所有部分。不要使用片段以外的信息补足缺失答案。'}


def exploratory_routing_schema():
    return {'task':'routing','labels':['search','code','writing','clarify'],
            'definitions':{
                'search':'完成明确任务所必需的信息缺失、需要查证实时信息或外部来源，应先搜索。',
                'code':'已有足够信息，下一步需要编写、运行或调试代码以完成交付。',
                'writing':'已有足够信息，下一步主要产出、整理或修改自然语言文本。',
                'clarify':'缺少决定任务或工具的关键条件，或任务不在给定工具能力范围内，应先澄清。'},
            'instructions':'决定任务的第一步。环境只有搜索、代码执行与文本写作三类工具；不能假设已经检索过最新资料。先澄清关键歧义，其次在必要外部信息不足时搜索，否则根据实际交付选择代码或写作。复合任务判断先做的步骤，而非最终成果。当前路由数据仍待真人独立标注，此定义尚不构成已验收正式任务。'}
