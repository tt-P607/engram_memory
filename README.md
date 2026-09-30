# Engram Memory

Engram Memory vNext 将聊天中的经历整理为有来源、可检索、可修订的正式记忆。当前插件入口只装配 vNext 组件。

## 工作方式

1. **捕捉经历**：群聊、私聊默认都收集，可分别通过 `vnext.candidate_encoder.group_enabled` 和 `private_enabled` 关闭。新消息按每个聊天流分别累计，达到 `message_threshold`（默认 30 条），或最早未处理消息等待达到 `max_wait_minutes`（默认 45 分钟）后，编码任务将消息交给配置的内部模型生成候选，并保存实际引用的来源消息。时间触发还要等待定时巡检，不保证恰好在第 45 分钟执行。关闭某类收集不会删除已有候选、正式记忆或来源，也不关闭闪回。
2. **Sleep 整理**：系统每天按 `vnext.sleep.daily_time`（默认本地时间 04:30）整理待处理及暂缓候选；本轮名单会分批全部处理，每批最多 100 条。积压达到压力阈值且安静窗口允许时也可能提前运行，启动时同样遵守安静窗口。Sleep 按相关人物、主题和时间组织候选，查阅来源、已存在的正式记忆及当天主动写入或修订的记忆，再决定新建、补充、修订、合并、关联、暂缓或忽略。执行后回读结果并交给模型复核，每组调查和复核共最多 8 次模型调用。候选生成本身不会直接创建正式记忆。
3. **回忆与使用**：`memory_search` 在不同聊天流之间检索正式记忆，支持人物、类型和时间筛选；`memory_read` 可读当前内容、版本历史或完整证据。`memory_write` 用于主动保存；当前聊天的 `memory_revise` 用于明确纠错，Sleep 也可依据查证结果进行修订。`person_lookup` 返回人物印象和相关近期记忆。回复前也可尝试自然闪回，但每次仍受相关性、耗时预算和冷却条件限制。

## 正式记忆结构

- **Memory**：正式记忆的稳定身份、创建时间和当前状态。
- **Revision**：按版本保存标题、正文、记忆类型、主体及参与人物；修订历史会保留，可回看旧版本。
- **Evidence**：记录来源类型，并关联旧记录或已保存的消息副本，用于查证记忆依据。

人物印象直接保存到核心数据库的 `PersonInfo.impression`，插件不另建人物印象表，也不保存另一份当前印象。Sleep 整理正式记忆后才审查印象，记录的只是审查理由、内容摘要和正式记忆关联。主动写入或修订正式记忆同样会在每日整理时触发印象审查；印象过长时让模型重新凝练完整正文。人物印象不代替正式记忆或其来源证据。

印象审查核对目标人物自己的来源发言，参与者身份不会把其他人的经历变成此人的经历。缺少可核对来源的旧画像仍完整保留为旧记忆，但不用于推断新的性格或经历；同一天的少量发言不足以形成稳定人格判断。

主动写记忆时，模型需要选择当前聊天中直接支持正文的准确消息 ID；不知道 ID 时，工具返回可选来源目录。只有实际引用的消息会保存证据副本，不能拿“帮我记住”这句话替代之前的具体陈述。

## 管理页

插件加载后提供本地管理页 `/engram-memory`，可查看候选、正式记忆、人物印象、来源和 Sleep 整理记录。Sleep 会话详情支持“可读”和“原始”视图。页面仅展示实际保存的过程日志和数据库操作；如果历史会话没有保存完整过程，会标明缺失，并只显示可核实的操作与回读结果。

## 配置与数据

插件默认启用。可在插件配置中调整以下 vNext 项：

| 配置项 | 默认值 | 用途 |
| --- | --- | --- |
| `internal_llm.task_name` | `tool_use` | 候选编码、Sleep 和人物印象使用的模型任务 |
| `vnext.candidate_encoder.group_enabled` | `true` | 收集群聊消息并编码候选 |
| `vnext.candidate_encoder.private_enabled` | `true` | 收集私聊消息并编码候选 |
| `vnext.candidate_encoder.message_threshold` | `30` | 达到该数量后编码新消息 |
| `vnext.candidate_encoder.max_wait_minutes` | `45` | 未达消息数量时的时间触发阈值，实际执行还需等待巡检 |
| `vnext.sleep.daily_time` | `04:30` | 每日 Sleep 整理时间，使用本地时区 |
| `vnext.candidate_encoder.pending_limit` | `1000` | 未处理候选超过上限时暂停新编码，包含暂缓和失败；历史隔离边界之前的积压不计入 |
| `vnext.sleep.batch_size` | `100` | 每批最多处理的候选数；每日本轮名单分批全部处理 |
| `vnext.prompt_injection.reminder_at_end` | `true` | 将记忆使用指引动态放在最新一轮输入；关闭后固定在首轮输入 |
| `vnext.flashback.enabled` | `true` | 开启自然闪回 |
| `vnext.flashback.trigger_probability` | `0.25` | 每轮回复尝试闪回的概率；`0` 关闭，`1` 每轮尝试 |
| `vnext.flashback.max_memories` | `2` | 每轮闪回最多注入条数，可设为 `0`、`1`、`2` |
| `vnext.flashback.cooldown_turns` | `3` | 同一记忆在同一聊天中的重复闪回间隔 |

正式记忆数据库默认位于 `data/engram_memory/vnext.db`，向量索引默认位于 `data/engram_memory/chroma`。可通过 `storage.vnext_db_path` 和 `storage.vector_db_path` 调整位置。

启用后，插件会注册 `memory_search`、`memory_read`、`memory_write`、`memory_revise`、`person_lookup` 等工具，并安排消息编码、后台整理及向量索引维护。自然闪回可通过 `vnext.flashback.enabled` 关闭。

记忆使用指引和闪回通过框架原生 `SystemReminder` 注入，不写入“附加上下文”。指引默认使用 `dynamic + forever`，随最新输入刷新；闪回使用 `fixed + forever`，放在当前聊天的首条 `USER` 中并持续保留。不同记忆使用独立名称，同一记忆不会反复累加；后续没有新闪回也不会删除已经想起的记忆。正式记忆更正后，已有提醒随之更新；记忆删除或来源隐私删除后移除对应提醒，读取失败时暂时撤回并在恢复后重建。上下文轮换时可从仍在的流提醒源重新注入。提醒源在插件卸载后清理，当前原生 Store 是内存存储，不保证跨 Bot 重启保存闪回念头；正式记忆库不受影响。提醒不会新增 `SYSTEM` 角色，也不会修改模型权重。触发概率对每个聊天的同一回复轮次只判断一次；未触发时不调用向量模型，即使触发也仍需满足相关性、冷却和耗时预算。旧顶层配置已移除，闪回配置统一在 `[vnext.flashback]`。

向量索引在正式记忆变更后立即尝试更新，后台每 10 秒检查待投递项。达到失败重试上限后停止自动重试，需明确执行恢复操作，避免无休止消耗模型额度。

## 迁移已有 Engram 数据结构

Schema 3 移除了插件自建人物印象表。已有正式记忆、历史版本、来源及核心数据库中的人物印象均保留。启动不会自动迁移旧库。

先制作包含 WAL 中最新数据的一致 SQLite 快照，放入独立目录；迁移输入须是已经合并 WAL、没有 WAL/SHM 文件的快照，不能只复制主文件而遗漏尚未合入的数据。再从项目根目录运行：

```powershell
uv run --no-sync python plugins/engram_memory/scripts/migrate_schema.py --source data/memory-copy/vnext.db --target data/memory-migrated/vnext.db
```

脚本要求交互确认，拒绝当前配置正在使用的数据库和已存在的输出文件。它只写新副本，并核对来源未变、其他数据摘要、数据库完整性及外键。v1 旧结构同时简化冗余字段；v2 迁移只移除插件人物表，不读取或覆盖核心人物印象。向量目录应另行复制并与新数据库一起切换；迁移脚本不会操作 Bot、修改配置或触碰桌面备份。

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
