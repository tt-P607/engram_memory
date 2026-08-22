"""engram_memory 插件配置。"""

from __future__ import annotations

from typing import ClassVar

from src.app.plugin_system.base import BaseConfig, Field, SectionBase, config_section


class EngramMemoryConfig(BaseConfig):
    """Engram Memory 三层记忆 + 人物连接配置模型。"""

    name: ClassVar[str] = "config"
    description: ClassVar[str] = "Engram Memory 三层记忆（短期/中期/长期）+ 人物连接"

    @config_section("plugin", title="插件设置", tag="plugin")
    class PluginSection(SectionBase):
        """插件级开关。"""

        enabled: bool = Field(
            default=True,
            description="插件总开关",
            label="启用插件",
            tag="plugin",
        )

    @config_section("storage", title="存储配置", tag="database")
    class StorageSection(SectionBase):
        """存储路径配置。"""

        metadata_db_path: str = Field(
            default="data/engram_memory/memory.db",
            description="记忆元数据库路径（SQLite）",
            label="元数据库路径",
            input_type="text",
            tag="file",
        )
        vector_db_path: str = Field(
            default="data/engram_memory/chroma",
            description="向量数据库路径（ChromaDB）",
            label="向量库路径",
            input_type="text",
            tag="file",
        )

    @config_section("retrieval", title="检索配置", tag="ai")
    class RetrievalSection(SectionBase):
        """语义检索参数。"""

        deduplication_threshold: float = Field(
            default=0.92, ge=0.0, le=1.0, description="检索结果去重阈值"
        )
        epa_skip_short_term: bool = Field(
            default=True, description="短期层检索是否跳过 EPA 重塑（记忆量小直接向量检索）"
        )
        base_beta: float = Field(default=0.3, description="EPA 重塑基础强度")
        logic_depth_scale: float = Field(default=0.5, description="逻辑深度对 beta 的放大系数")
        core_boost_min: float = Field(default=1.2, description="核心标签增益下限")
        core_boost_max: float = Field(default=1.4, description="核心标签增益上限")
        diffusion_boost: float = Field(default=0.3, description="扩散标签增益")
        opposing_penalty: float = Field(default=0.5, description="对立标签惩罚")

    @config_section("write_conflict", title="写入判重配置", tag="ai")
    class WriteConflictSection(SectionBase):
        """记忆写入时的新颖度判重参数。"""

        top_n: int = Field(default=8, ge=1, le=50, description="写入时检索的邻域向量数量")
        energy_cutoff: float = Field(
            default=0.1, ge=0.0, le=1.0, description="新颖度能量比阈值，低于则视为重复"
        )

    @config_section("short_term", title="短期记忆配置", tag="timer")
    class ShortTermSection(SectionBase):
        """短期记忆层参数。"""

        enabled: bool = Field(
            default=True,
            description="是否启用短期记忆后台总结与注入",
            label="启用短期记忆",
            tag="timer",
        )
        ttl_hours: int = Field(default=48, ge=1, le=168, description="短期记忆 TTL（小时）")
        summarizer_interval_minutes: int = Field(
            default=30, ge=5, le=180, description="短期总结周期（分钟）"
        )
        summarizer_message_threshold: int = Field(
            default=20, ge=5, le=100, description="触发短期总结所需的新增消息数"
        )
        inject_threshold: float = Field(
            default=0.75, ge=0.0, le=1.0, description="短期记忆被动注入相似度阈值"
        )
        inject_max: int = Field(default=3, ge=1, le=10, description="短期记忆被动注入最大条数")
        max_items: int = Field(
            default=500, ge=50, le=5000, description="短期记忆数量上限，超过时清理最旧的"
        )

    @config_section("flashback", title="记忆闪回", tag="ai")
    class FlashbackSection(SectionBase):
        """记忆闪回注入参数。"""

        enabled: bool = Field(default=True, description="是否启用记忆闪回")
        trigger_probability: float = Field(
            default=0.25, ge=0.0, le=1.0, step=0.01, input_type="slider",
            description="闪回触发概率", depends_on="enabled", depends_value=True,
        )
        gray_zone_min: float = Field(
            default=0.55, ge=0.0, le=1.0, description="闪回灰色地带下界",
            depends_on="enabled", depends_value=True,
        )
        gray_zone_max: float = Field(
            default=0.70, ge=0.0, le=1.0, description="闪回灰色地带上界",
            depends_on="enabled", depends_value=True,
        )
        candidate_limit: int = Field(
            default=50, ge=10, le=200, description="闪回候选数量上限",
            depends_on="enabled", depends_value=True,
        )
        activation_weight_exponent: float = Field(
            default=1.0, ge=0.5, le=3.0, step=0.1,
            description="闪回权重选择激活指数",
            depends_on="enabled", depends_value=True,
        )
        cooldown_seconds: int = Field(
            default=3600, ge=0, le=86400, description="闪回冷却期（秒）",
            depends_on="enabled", depends_value=True,
        )

    @config_section("journal", title="人物印象", tag="ai")
    class JournalSection(SectionBase):
        """人物印象参数（日记回顾已移除，仅保留印象相关配置）。"""

        impression_max_chars: int = Field(
            default=600, ge=100, le=2000, description="人物印象最大字数"
        )

    @config_section("internal_llm", title="内部 LLM 配置", tag="ai")
    class InternalLLMSection(SectionBase):
        """内部子代理使用的模型任务配置。"""

        task_name: str = Field(
            default="tool_use", description="内部子代理模型任务名（chat）"
        )
        embedding_task_name: str = Field(
            default="embedding", description="内部 embedding 模型任务名"
        )

    @config_section("persona", title="人物蒸馏", tag="ai")
    class PersonaSection(SectionBase):
        """人物印象懒加载蒸馏参数。"""

        enabled: bool = Field(
            default=True,
            description="是否启用人物印象懒加载蒸馏",
            label="启用人物蒸馏",
            tag="ai",
        )
        scope: str = Field(
            default="all",
            description="印象蒸馏的消息范围：all=全部 / group=仅群聊 / private=仅私聊（可按人物覆盖）",
        )
        min_messages: int = Field(
            default=200, ge=20, le=10000,
            description="触发蒸馏所需的最少本人文本消息条数",
        )
        max_messages: int = Field(
            default=2000, ge=100, le=50000,
            description="锚点定位时最多取用的本人消息条数（取最新）",
        )
        window_count: int = Field(
            default=15, ge=3, le=60,
            description="蒸馏采样的对话窗口数量",
        )
        window_radius: int = Field(
            default=12, ge=2, le=50,
            description="每个窗口在锚点前后各取的消息条数",
        )
        chunk_size: int = Field(
            default=200, ge=50, le=2000,
            description="分块提炼时每块的文本消息条数",
        )
        memory_index_limit: int = Field(
            default=100, ge=10, le=300,
            description="印象蒸馏选记忆环节展示的记忆目录条数",
        )
        memory_material_limit: int = Field(
            default=20, ge=3, le=50,
            description="印象蒸馏允许取用的记忆全文条数",
        )

    plugin: PluginSection = Field(default_factory=PluginSection)
    storage: StorageSection = Field(default_factory=StorageSection)
    retrieval: RetrievalSection = Field(default_factory=RetrievalSection)
    write_conflict: WriteConflictSection = Field(default_factory=WriteConflictSection)
    short_term: ShortTermSection = Field(default_factory=ShortTermSection)
    flashback: FlashbackSection = Field(default_factory=FlashbackSection)
    journal: JournalSection = Field(default_factory=JournalSection)
    internal_llm: InternalLLMSection = Field(default_factory=InternalLLMSection)
    persona: PersonaSection = Field(default_factory=PersonaSection)
