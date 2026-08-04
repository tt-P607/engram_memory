# Engram Memory（记忆痕迹）

Neo-MoFox 三层记忆插件：短期 / 中期 / 长期记忆 + 人物连接。

## 功能

- **三层记忆**：短期（48h TTL）→ 中期（活跃缓冲）→ 长期（永久归档），跨群全局检索
- **人物连接**：通过 `person_lookup` 查询人物认知（昵称 / 印象 / 交互时间线）与相关记忆索引
- **人物印象懒加载蒸馏**：基于人物本人跨流文本消息（攒够 1000 条）自动蒸馏印象，懒加载 + 夜间巡检触发，不浪费 LLM
- **记忆闪回**：基于语义关联的概率性联想注入
- **管理后台**：Web 页面 + REST API，可查看 / 编辑 / 删除记忆

## 组件

| 类型 | 名称 | 说明 |
|------|------|------|
| Tool | `memory_search` | 语义检索记忆，支持按层 / 人物 / 标签过滤 |
| Tool | `memory_read` | 按 ID 批量读取记忆全文 |
| Tool | `memory_write` | 创建或更新记忆 |
| Tool | `memory_delete` | 软删除单条记忆 |
| Tool | `person_lookup` | 查询人物认知 + 记忆索引目录 |
| EventHandler | `short_term_injector` | 短期记忆被动注入（DYNAMIC） |
| EventHandler | `flashback_injector` | 记忆闪回注入（DYNAMIC） |
| EventHandler | `private_chat_person_injector` | 私聊人物认知注入（FIXED） |
| Router | `memory_admin` | 管理后台 `/engram-memory` |

## 功能开关

以下开关默认开启，可在插件配置（`config/plugins/engram_memory/config.toml`）中关闭：

| 配置 | 作用 |
|------|------|
| `plugin.enabled` | 插件总开关 |
| `short_term.enabled` | 短期记忆后台总结与短期注入 |
| `persona.enabled` | 人物印象懒加载蒸馏 |
| `flashback.enabled` | 记忆闪回 |

关闭后相应后台任务 / 注入不再运行，但 6 个记忆工具仍可被 LLM 显式调用。

## 安装

将 `plugins/engram_memory` 目录放入 Neo-MoFox 的 `plugins/` 目录，重启应用即可自动加载。

## 配置

配置示例见 `config/plugins/engram_memory/config.toml`，主要节：

- `storage`：元数据库 / 向量库 / 日记存储路径
- `short_term`：TTL、总结周期、注入阈值与条数、数量上限
- `journal`：触发小时、启动补偿阈值、活跃流判定、印象字数上限
- `flashback`：闪回概率、灰色地带、冷却期
- `internal_llm`：内部子代理使用的模型任务名

## 数据存储

| 数据 | 位置 |
|------|------|
| 记忆元数据 | SQLite（`data/engram_memory/memory.db`） |
| 记忆向量 | ChromaDB（`data/engram_memory/chroma`） |
| 日记 | Markdown 文件（`data/engram_memory/journals`） |

## 开发

```bash
ruff check plugins/engram_memory/   # 代码检查
pytest test/plugins/engram_memory/  # 运行测试
```
