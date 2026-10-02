# Engram Memory

Engram Memory 保存可追溯、可检索、可修订的正式记忆，并根据正式记忆变化更新核心人物印象。

## 工作方式

1. **保存正式记忆**：`memory_write` 创建记忆，`memory_revise` 保留旧版本并记录修订，`memory_invalidate` 作废不再有效的记忆。每条记忆保留人物关联、来源与稳定 ID。
2. **更新人物印象**：正式记忆提交后发布变化事件。Persona 更新流程按受影响人物合并任务，并依据关联正式记忆维护核心人物印象。
3. **检索与回忆**：`memory_search` 检索正式记忆，`memory_read` 查看当前内容、版本历史和证据。`person_lookup` 返回人物印象与近期相关记忆。回复前可尝试自然闪回，仍受相关性、耗时预算和冷却条件限制。

## 正式记忆结构

- **Memory**：正式记忆的稳定身份、创建时间和当前状态。
- **Revision**：按版本保存正文、记忆类型、主要人物和次要人物；修订历史会保留，可回看旧版本及其来源。
- **Evidence**：记录来源类型，并关联旧记录或已保存的消息副本，用于查证记忆依据。

人物印象直接保存到核心数据库的 `PersonInfo.impression`，插件不另建人物印象表，也不保存另一份当前印象。记忆变化会触发受影响人物的印象更新；人物印象不代替正式记忆或其来源证据。

每条记忆有一个主要人物，可有关联的次要人物。人物 ID 必须来自核心人物查询，不能用昵称、猜测的账号或自行拼接的标识替代。主要人物是记忆最直接描述的人，不必是来源发送者；被提及但没有亲自发言的人也可以关联，只要身份已核实且正文与来源支持该关联。人物关联不表示本人确认了陈述，也不能把其他人的经历归给此人。

人物印象请求直接提供该人物全部当前有效关联记忆的完整标题和正文，无需模型调用工具读取；同时注入完整人设，复用核心 `actor` 模型任务，不设正文字符上限。模型以 `[Memory: UUID]` 引用真实记忆 ID，程序校验依据后，将相邻引用合为正文圈号，完整 ID 放在末尾的“记忆依据”中；相同依据组复用编号，一条尾注可对应多条记忆。圈号覆盖 1 到 50，更多尾注使用 `[51]` 等编号，不限制引用数量。再次更新前，程序将尾注还原为行内引用；已有行内引用格式在印象更新时转换，不批量重写历史印象。可通过 `memory_read` 核对依据；转述、计划和不确定判断保留其证据边界，少量观察不足以证明稳定人格。印象本身不作为记忆证据。

## 写入与来源

写入前先通过 `memory_search` 检索同一人物、主题或经历，再回读已有正文，避免重复。新发生的独立事实创建新记忆；对已有记录的纠正或补充创建修订，不覆盖旧版本。正文由 Bot 根据来源组织为完整、自然的叙述，区分转述者、原陈述者、计划与实际发生的事实，不把猜测改写成确定结论。

| Action | 参数 |
| --- | --- |
| `memory_write` | payload 必填 `content`、`memory_kind`、`primary_person_id`、`source_message_ids`；可选 `secondary_person_ids`。 |
| `memory_revise` | payload 必填 `memory_id`、`based_on_revision_id`、`content`、`primary_person_id`、`source_message_ids`；可选 `secondary_person_ids`、`memory_kind`、`reason`。版本 ID 使用 `memory_read` 回读的当前版本。 |
| `memory_invalidate` | `memory_id`、`reason`、`source_message_ids`；作废保留历史版本与证据。 |

`source_message_ids` 选择当前聊天中直接支持正文的真实消息 ID，并包含理解回复、问答或指代所需的来源；按接口返回的来源目录核对，不能猜测或伪造。只有实际引用的消息保存为证据快照，来源聊天、消息身份、原文和消息时间由程序记录。“帮我记住”的指令不能替代具体事实的来源。历史导入记忆保留原记录及来源类型，不补造未保存的消息。

## 管理页

插件加载后提供管理页 `/engram-memory`，包含概览、正式记忆和人物印象三个页面。管理数据 API 只接受回环地址请求。可查看状态、当前正文、历史版本、主要及次要人物的核心 ID、消息来源和核心印象；印象中的 Memory ID 可打开记忆详情。

Doctor 提供正式记忆、版本证据、检索入口和向量索引的检查与派生修复。历史候选独享的消息证据不纳入正式版本的快照完整性检查；修复不删除正式记忆、旧候选、整理会话或来源数据。

## 配置与数据

插件默认启用。可在插件配置中调整以下 vNext 项：

| 配置项 | 默认值 | 用途 |
| --- | --- | --- |
| `vnext.persona.recent_memory_limit` | `10` | 人物查询返回的近期记忆条数 |
| `vnext.retrieval.default_limit` | `5` | 默认检索返回条数 |
| `vnext.retrieval.max_limit` | `20` | 检索返回条数上限 |
| `vnext.retrieval.rrf_k` | `60` | 多路召回的排名融合参数 |
| `vnext.prompt_injection.reminder_at_end` | `true` | 将记忆使用指引动态放在最新一轮输入；关闭后固定在首轮输入 |
| `vnext.flashback.enabled` | `true` | 开启自然闪回 |
| `vnext.flashback.trigger_probability` | `0.25` | 每轮回复尝试闪回的概率；`0` 关闭，`1` 每轮尝试 |
| `vnext.flashback.context_turns` | `6` | 闪回检索使用的上下文轮数 |
| `vnext.flashback.latency_budget_ms` | `1200` | 闪回检索的耗时预算，单位毫秒 |
| `vnext.flashback.max_memories` | `2` | 每轮闪回最多注入条数，可设为 `0`、`1`、`2` |
| `vnext.flashback.cooldown_turns` | `3` | 同一记忆在同一聊天中的重复闪回间隔 |
| `vnext.vector.worker_retry_limit` | `3` | 向量任务的失败重试上限 |

正式记忆数据库默认位于 `data/engram_memory/vnext.db`，向量索引默认位于 `data/engram_memory/chroma`。可通过 `storage.vnext_db_path` 和 `storage.vector_db_path` 调整位置。

启用后，插件注册三个正式记忆 Action、三个查询 Tool、记忆变化与闪回 EventHandler、记忆 Service、Doctor 与管理 Router，并维护派生向量索引。配置的 vnext 节仅包含 `persona`、`retrieval`、`flashback`、`prompt_injection`、`vector`。自然闪回可通过 `vnext.flashback.enabled` 关闭。

记忆使用指引和闪回通过框架原生 `SystemReminder` 注入，不修改模型权重。指引默认随最新输入刷新；闪回按记忆 ID 管理，同一记忆不会反复累加。正式记忆更正或作废时，已有提醒随之更新或移除；来源隐私删除也会撤回对应提醒。提醒源在插件卸载时清理，不保证跨 Bot 重启保留；正式记忆库不受影响。未触发闪回时不调用向量模型，触发后仍受相关性、冷却和耗时预算限制。

正式记忆变更产生的向量待投递项由后台任务处理，轮询间隔为 10 秒；记忆变化事件的发布不直接更新向量索引。达到失败重试上限后停止自动重试，需明确执行恢复操作，避免无休止消耗模型额度。

正式记忆、历史版本、来源证据、数据库中的候选与整理会话记录、导入资料和现有备份均保留。这些数据独立于运行配置保存。配置加载只过滤本地遗留的废弃字段，不重写配置文件，其他未知字段仍接受严格校验。

## 迁移已有 Engram 数据结构

当前 Schema 将人物印象保存在核心数据库。已有正式记忆、历史版本、来源及核心人物印象均保留；启动不会自动迁移不兼容的数据库。

先制作包含 WAL 中最新数据的一致 SQLite 快照，放入独立目录；迁移输入须是已经合并 WAL、没有 WAL/SHM 文件的快照，不能只复制主文件而遗漏尚未合入的数据。再从项目根目录运行：

```powershell
uv run --no-sync python plugins/engram_memory/scripts/migrate_schema.py --source data/memory-copy/vnext.db --target data/memory-migrated/vnext.db
```

脚本要求交互确认，拒绝当前配置正在使用的数据库和已存在的输出文件。它只写新副本，并核对来源未变、其他数据摘要、数据库完整性及外键。支持的来源结构为 Schema v1 和 v2，不读取或覆盖核心人物印象。向量目录应另行复制并与新数据库一起切换；迁移脚本不会操作 Bot、修改配置或触碰备份。

## 从 Booku 迁移记忆

迁移器只把 Booku 的 `memory` bucket 导入 Engram；`knowledge` 文档片段会明确计数并跳过，不会变成正式记忆。迁移不会写入当前 Engram 或 Booku 数据库，也不会修改配置。目标目录必须是新目录，已存在时拒绝覆盖。

在项目根目录运行：

```powershell
uv run --no-sync plugins/engram_memory/scripts/migrate_booku.py
```

默认先显示 Booku 配置所指元数据库中的记忆数量，并说明知识片段将被跳过。只有在提示处准确输入 `迁移`，才会创建新目标目录并执行导入、重建向量索引、回读和实际检索核对；其他输入不会开始迁移。`--preview` 只检查并显示预览后退出。没有 `--apply` 或 `--yes` 这类跳过交互确认的选项。

可选参数：

| 参数 | 行为 |
| --- | --- |
| `--source` | 指定 Booku SQLite 元数据库的相对路径；省略时读取 Booku 配置中的路径。 |
| `--output` | 指定全新的相对输出目录；默认 `data/engram_memory/booku-import`。 |
| `--batch-size` | 每次向量模型请求的记忆条数；默认 `32`，上限 `100`。 |
| `--resume` | 仅续跑本迁移器已完整导入、但向量索引失败的私有输出目录；仍需交互输入 `继续`。 |

每条旧记忆成为一条正式记忆及其初始版本，保留原记录创建时间；Booku 没有保存的旧版本不会补造。已软删除的记录写为 `TOMBSTONED`，归档记忆仍可使用。临时备忘不导入为正式记忆，只留在源库快照中。缺失、异常或占位的人物标识不会靠昵称推测；旧来源只记为 `LEGACY_RECORD`，不会伪造消息证据。

类型映射为 `event → EVENT`、`preference → PREFERENCE`，其余旧记忆类型进入 `FACT`；旧分类、标签和原始字段完整留在来源记录中。筛选依据是 `bucket`：记忆区中原类型为 `knowledge` 的记录仍是记忆，会保留；知识库 `bucket=knowledge` 的文档片段不会导入。迁移不让模型改写旧正文。

完成后，新目录包含 `vnext.db`、`chroma`、Booku SQLite 原始副本，以及供本机核对的简明 `report` 和 `events`。原始副本是完整 SQLite 快照，包含源库中的所有内容。迁移输出可能含私人资料或凭据，仅供本机核对，请勿对外分享。迁移本身不切换正在运行的 Engram。需要启用新库时，先停止 Bot，再手动将 `storage.vnext_db_path` 与 `storage.vector_db_path` 成对改到新目录；不会自动合并或覆盖现有 Engram 记忆，本说明中的迁移步骤也不会替你切换配置。
