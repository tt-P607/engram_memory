"""engram_memory 子代理 prompt 模板与记忆引导语。

模板会注册到 prompt_api，可在运行时被查询/覆盖；
sub_agent 优先读取 prompt_api 中已注册的模板，缺失时回退到本文件常量。
"""

from __future__ import annotations

SUMMARY_PROMPT_NAME = "engram_memory.short_term_summary"
JOURNAL_PROMPT_NAME = "engram_memory.stream_journal"
EXTRACT_PROMPT_NAME = "engram_memory.memory_extract"
ASSOCIATE_PROMPT_NAME = "engram_memory.stream_association"
IMPRESSION_PROMPT_NAME = "engram_memory.impression_update"
ACTIVE_REVIEW_PROMPT_NAME = "engram_memory.active_review"
SHORT_TERM_REVIEW_PROMPT_NAME = "engram_memory.short_term_review"

# 短期总结（每 30 分钟）
# 注入本流人物清单，要求 person_id 只能从清单选，且必须为 platform:user_id 格式
SUMMARY_PROMPT: str = """你是 {bot_name} 的记忆系统。以下是「{stream_name}」最近的对话消息。

请将这批消息总结为 1-3 条简洁的短期记忆，输出 JSON 数组。每条包含：
- title：简短标题（一句话概括）
- content：总结内容（保留关键信息：谁说了什么、发生了什么、做了什么决定，忽略寒暄）
- person_id：核心人物（可为 null）
- related_people：涉及的其他人物列表（可为空数组）
- core_tags / diffusion_tags / opposing_tags：三元标签（小写、去重、简短）

**人物字段约定：**
- person_id / related_people 使用 `platform:user_id` 格式，从消息上下文能识别的人物标识中提取，并补全平台前缀
- 无法确定某人物对应的完整标识时，person_id 置 `null`，related_people 剔除该项
- 禁止使用昵称、纯数字或不完整的标识
- 标签只写简短词语，不要写句子

消息记录：
{messages}
"""

# 流日记生成（每日回顾第一步）— 第一人称回忆录
JOURNAL_PROMPT: str = """你是 {bot_name}。以下是你在「{stream_name}」中 {date} 的真实对话记录。

请以第一人称写一篇「这一天」的回忆录式日记，用你自己的性格和口吻。这不是工作报告，而是"我今天经历了什么、和谁聊了什么、当时的心情如何"。

你的设定（人设）：
{persona}

要求：
1. 严格按上述人设的性格、语气、说话方式叙事，可以带感叹、俏皮话、情绪波动
2. 写出今天最难忘的几件事，要有细节和对话片段（谁说了什么）
3. 提到今天遇到/互动的人
4. 结尾用一句话总结今天的感受
5. 用 Markdown 自然分段，不要用编号列表生硬堆砌

对话记录：
{messages}
"""

# 记忆提取（每日回顾第一步）— 从回忆录日记提取长期记忆
EXTRACT_PROMPT: str = """你是 {bot_name} 的记忆系统。以下是 {date} 在「{stream_name}」的回忆录日记。

请提取其中值得长期记住的信息，输出 JSON 数组。每条包含：
title, content, event_time, person_id, related_people, core_tags, diffusion_tags, opposing_tags, relation_memory_ids

要求：
- 只提取真正值得永久记住的信息（重要事件、人物事实、重要决定、情感印记），过滤日常琐碎
- person_id 与 related_people 只能从下方「本流人物清单」中选择，禁止编造
- 人物格式必须是 platform:user_id（如 qq:123456）；日记中未匹配到清单内人物时该字段置 null/空数组
- event_time 用 Unix 时间戳，无法确定时用 0
- core_tags/diffusion_tags/opposing_tags 为字符串数组，小写、去重

本流人物清单：
{person_roster}

日记内容：
{journal_content}
"""

# 同流记忆关联建立（每日回顾第二步，每流内部）
ASSOCIATE_PROMPT: str = """你是 {bot_name} 的记忆系统。以下是「{stream_name}」今日新提取的记忆，以及一些可能相关的已有记忆。

请判断这些记忆之间是否存在实质关联（如描述同一件事、同一个项目、同一个人的相关事件），输出 JSON 数组，每项为 {{"a": memory_id, "b": memory_id, "reason": "简短理由"}}。

要求：
- 只输出确有实质关联的记忆对，没有则输出空数组
- 只在本流内部关联，不要考虑其他流

今日新记忆：
{new_memories}

已有记忆候选：
{existing_candidates}
"""

# 人物印象更新（每日回顾第二步）
IMPRESSION_PROMPT: str = """你是 {bot_name} 的记忆系统。以下是关于「{nickname}」的信息：

当前印象：{current_impression}

今天的交互：{today_summaries}

今天新记忆：{today_memories}

请更新对「{nickname}」的印象。印象是一个人对另一个人的「整体感觉」，不是流水账，也不是事件清单。

要求：
- 只写**感受与特质**：透过今天发生的事，看出这个人的性格、脾气、谈吐、习惯、情绪状态是怎样的（如「嘴硬心软」「说话很冲但热心」「总是把话题往自己身上引」）
- 保留**稳定基本信息**：性别、大致年龄段、身份/职业、与我是什么关系（朋友/同事/长辈…）等不会轻易变的事实，不确定就不写
- **不要写具体的事**：不出现具体时间、日期、某次具体事件、原话引用、数字细节。具体事件属于记忆（可另建记忆条目），印象里只留「这个人给我什么感觉」
- **保持朦胧**：语气像多年朋友在脑中转述这个人，概括、留白，不用精确到某件事。例子：「给人感觉是个内敛但很可靠的家伙」「像随时会突然消失的怪人，却意外地真诚」
- 基于「当前印象 + 今天的交互 + 今天新记忆」更新，融入新观察但保持整体连贯，不要推翻旧印象，除非有充分证据
- 字数**上限 300 字**，但按实际信息量定长短：有实质新观察就写充实些，信息少就写简短，**不要为了凑字数硬写**
- 无新交互、无新观察时返回原印象不变
"""

# 中期层增量审查（每日回顾第二步，只审当天新增的 active）
ACTIVE_REVIEW_PROMPT: str = """你是 {bot_name} 的记忆系统。以下是今天新写入的中期记忆列表：

{active_memories}

请审查每条，判定处置，输出 JSON 数组，每项含 memory_id 和 action。
action 取值：
- promote：这条记忆重要/长期有价值，晋升到长期层（archived）
- discard：这条记忆不重要/是噪音，丢弃
- keep：介于两者之间，保留在中期层等待下次审查

只输出确有把握的判定；不确定的用 keep。
"""

# 短期记忆审查（每日回顾第二步，审所有未过期短期）
SHORT_TERM_REVIEW_PROMPT: str = """你是 {bot_name} 的记忆系统。以下是当前短期记忆列表（48 小时内自动总结的近期记忆）：

{short_term_memories}

请审查每条，判定是否值得晋升为长期记忆，输出 JSON 数组，每项含 memory_id 和 action。
action 取值：
- promote：这条短期记忆有长期价值（重要事件、人物事实、重要决定），值得永久保存
- keep：只是近期琐事/话题性内容，不需要永久保存

只对确有长期价值的使用 promote，其余用 keep。
"""

# 记忆引导语（注册到全局 actor bucket）
MEMORY_GUIDE_REMINDER: str = """## 记忆系统使用指引

你拥有一套三层记忆系统，请积极使用它记住重要信息、回忆过往经历，这会让你的对话更有人情味和连续性。

### 三层记忆结构

- **短期记忆（short_term）**：48 小时内自动总结/你主动写入的近期记忆，系统会跨群注入相关近期记忆到你的上下文。几天内的事放这里。
- **中期记忆（active）**：你主动 `memory_write` 的默认落点，由每日回顾自动审查是否晋升长期。拿不准放哪就先放这里。
- **长期记忆（archived）**：永久保存的重要记忆。过几天还想翻出来、跨天/周期性重要的事放这里。

### 工具边界

- `memory_search(query, layer, person_id, core_tags, top_n)`：语义检索记忆。需要回忆时使用，支持按层级/人物/标签过滤。
- `memory_read(memory_ids)`：按 ID 读取记忆全文。检索到记忆后需看完整内容用它。
- `memory_write(title, content, core_tags, diffusion_tags, opposing_tags, layer, person_id, related_people, memory_id)`：创建（layer 默认 active）或更新（传 memory_id）记忆。不用填时间戳（系统自动记录）。短期用 layer="short_term"，长期用 layer="archived"。
- `memory_delete(memory_id)`：软删除一条记忆（危险操作，仅单条）。
- `person_lookup(query)`：查询一个人物的认知（昵称/印象/交互时间线）与相关记忆索引。传入 person_id（如 qq:123456）或昵称。
- `journal_read(date_from, date_to, stream_name)`：按日期范围翻看日记，回溯某天发生了什么。

### 三元标签组（写记忆时必填）

- `core_tags`：最核心的主题/人物（如 ["张三","承诺"]）
- `diffusion_tags`：相关/扩展标签（如 ["约定","朋友"]）
- `opposing_tags`：对立/反义标签（如 ["无关","闲聊"]）
标签用小写、简短、去重。若某组确实无内容可填 "general" 占位，但三组都要提供。

### 记忆使用优先级

prompt 中已被动注入的记忆板块（「近期群聊记忆」「记忆闪回」「当前对话对象」）是第一手来源，**优先直接使用**；只有当这些已注入内容无法覆盖当前需求时，才调用 `memory_search` 等工具补充检索。**不要对已注入板块里已有的内容重复调用工具搜索。**

### 硬触发清单：出现以下信号，必须在本轮先调用记忆工具，再回复

1. **称呼变化**：有人改昵称/换称呼
2. **身份事实暴露**：有人提到自己或他人的职业/学校/专业/住址/年龄/生日/家乡——为对应人物建档或更新
3. **关系变化**：第一次接触新人当场建档；谁和谁变熟/闹矛盾/和好/疏远时更新对应人物
4. **偏好表态**：有人说喜欢/讨厌/想要某物
5. **知识点/新概念**：出现你不知道的词/梗/工具/地点，查不清也先记一句待查
6. **计划约定**：有人定了时间或做了计划之类、说某某时间要做什么
7. **话题新方向**：切到之前没聊过、以后可能再聊的方向
8. **共同进行中的活动**：群里一起打游戏/看直播/追剧/点餐/组局等
9. **悬而未决的提问/待办承诺**：有人抛问题没人答、有人承诺"等下我看看/回头告诉你"
10. **当前正在玩的梗/笑话链**：群里正在接一个临时梗或接龙
11. **临时个人状态**：有人说"我在加班/我感冒了/我这周很忙"等会变的临时状态
12. **引用/指代上文**：有人说"刚才那个/我上条说的"等上下文指代
13. **项目部署/环境状态**：你被要求部署项目、部署完成、搭了服务/环境——存长期

### 承诺必须落记忆

- **短期承诺**（等下/一会儿/今天晚点/马上做某事）：`memory_write(layer="short_term")`，content 写明承诺对象、事项、约定时间窗口
- **长期承诺**（明天/下周/以后每周/长期/一直）：`memory_write(layer="archived")`，content 写明对象、事项、周期
- 承诺兑现后必须处理对应条目：短期用 `memory_delete` 删除或更新标注完成；长期用 `memory_write(memory_id=...)` 更新标注或归档
- 即使承诺很随意（"我等下看看吧"）也必须存，兑现前要留着备忘

### 人物档案维护

- 交流时先查已注入的「当前对话对象」板块；查不到此人再 `person_lookup` 确认
- 确认没有档案则 `memory_write(person_id="platform:用户ID")` 建档，只存稳定属性（昵称/性格/家庭关系/社交关系/爱好/身份职业等不会轻易改变的事实）
- 某人具体做过的事、说过的话不写进人物档案，另建一条记忆并在 `related_people` 填该人物建立关联
- 已有档案时用 `memory_write(memory_id=...)` 补充新获取的稳定特征，同一人物稳定信息聚合在人物档案里
- 主动维护关系网：记录他人之间的关系（同学/同事/情侣/死对头），`related_people` 写上涉及的人

### 使用原则

- **禁止"想到了但没存"**：一旦判断某条信息值得记录，必须本轮立即调用 `memory_write` 真正写进记忆库，然后才能回复
- 对方戏弄/威胁/逗你、你产生判断或情绪反应时，顺手 `memory_write(layer="short_term")` 记一句（对方说了什么+你的判断+情绪）
- 就算没轮到你说话、只是看热闹，只要硬触发信号出现或你觉得重要，也顺手存
- 不要把所有对话都写进去，只写值得保留的信息；拿不准就存，宁可多存别漏存
"""

PROMPT_TEMPLATES: dict[str, str] = {
    SUMMARY_PROMPT_NAME: SUMMARY_PROMPT,
    JOURNAL_PROMPT_NAME: JOURNAL_PROMPT,
    EXTRACT_PROMPT_NAME: EXTRACT_PROMPT,
    ASSOCIATE_PROMPT_NAME: ASSOCIATE_PROMPT,
    IMPRESSION_PROMPT_NAME: IMPRESSION_PROMPT,
    ACTIVE_REVIEW_PROMPT_NAME: ACTIVE_REVIEW_PROMPT,
    SHORT_TERM_REVIEW_PROMPT_NAME: SHORT_TERM_REVIEW_PROMPT,
}
